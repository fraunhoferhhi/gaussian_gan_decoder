#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from datetime import datetime
import os
import sys

import numpy as np
import torch

sys.path.append("./")
sys.path.append("../eg3d/eg3d/")
from gaussian_decoder.triplane_decoder import GaussianTriplaneDecoder
from utils.loss_utils import l1_loss, ssim
from eg3d_utils.plot_utils import make_3d_plot, log_to_wandb
from gaussian_renderer import network_gui, render_simple
import sys
from scene import GaussianModel
from scene import SceneEG3D_flame as SceneEG3D
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb

def training(opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    path = "./output"
    wandb.init(project='3DGaussianHeads', dir="./output", group="flame+decoder", name="init-test")
   
    new_z_interval = np.inf

    white_background = False
    first_iter = 0
    batch_size = 1
    gaussians = GaussianModel(3)
    eg3d_model = GaussianTriplaneDecoder(num_gaussians_per_axis=10, triplane_generator_ckp="../eg3d/eg3d/networks/var3-128.pkl", pre_offset=opt.pre_offset, hugs=False)
    scene = SceneEG3D(eg3d_model=eg3d_model, gaussians=gaussians, overfit_single_id=opt.overfit_single_id, flame_init=True)

    decoder_params = [
        {'params': eg3d_model.color_opacity_decoder.parameters(), 'lr': opt.rotation_lr, "name": "color_opacity_decoder"},
        {'params': eg3d_model.scaling_rotation_decoder.parameters(), 'lr': opt.rotation_lr, "name": "scaling_rotation_decoder"}    
        ]

    if opt.pre_offset:
        decoder_params += [{'params': eg3d_model.pre_offset_decoder.parameters(), 'lr': 0, "name": "pre_offset_decoder"}]


    gaussians.training_setup(opt, decoder_params=decoder_params)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # bg_color = [1, 1, 1]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=False)
    first_iter += 1

    # init scale rot
    scene.init_rot_scale_decoder(gaussians.optimizer, gaussians.get_xyz)

    gaussians.kill_all_but_xyz_learning_rate()
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        if opt.pre_offset:
            if opt.kill_xyz:
                if iteration < opt.pre_offset_start:
                    gaussians.update_learning_rate(iteration)
                elif iteration == opt.pre_offset_start:
                    gaussians.kill_xyz_learning_rate()
                    gaussians.set_pre_offset_learning_rate(opt.rotation_lr)
            else:
                gaussians.update_learning_rate(iteration)
                if iteration == opt.pre_offset_start:
                    gaussians.set_pre_offset_learning_rate(opt.rotation_lr)
        else:
            gaussians.update_learning_rate(iteration)

        if iteration % new_z_interval == 0:
            scene.z_list.append(torch.randn(1, 512).float())

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        fov_deg = np.random.rand() * 5 + 13 # (13 - 18)

        with torch.no_grad():
            _, gt_image, _ = scene.get_camera_and_target(fov_deg, xyz=gaussians.get_xyz)

            # get flame xyz from gt_image
            flame_xyz = scene.get_flame_verts(gt_image)
            flame_xyz = torch.nn.Parameter(flame_xyz)

        viewpoint, gt_image, decoded_features = scene.get_camera_and_target(fov_deg, xyz=flame_xyz, z=scene.z, extrinsic=scene.extrinsic)

        gaussians._features_dc = decoded_features["_features_dc"][0].unsqueeze(1)
        gaussians._scaling = decoded_features["_scaling"][0]
        gaussians._rotation = decoded_features["_rotation"][0]
        gaussians._opacity = decoded_features["_opacity"][0]
        gaussians._features_rest = torch.nn.Parameter(torch.zeros((gaussians._features_dc.shape[0], 15, 3)).cuda())
        gaussians._xyz = flame_xyz

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") #if opt.random_background else background

        render_pkg = render_simple(viewpoint, gaussians, bg_color=bg, xyz_offset=decoded_features["_xyz_offset"])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = image[:3]

        # Loss
        log_dict = {}
        gt_image = gt_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)    
        loss.backward()
        
        log_dict["Loss/L1"] = Ll1
        log_dict["Loss/SSIM"] = 1.0 - Lssim
        log_dict["Loss/total"] = loss

        iter_end.record()

        with torch.no_grad():
            if iteration % 10 == 0 or iteration == 1:
                image = torch.concat([gt_image, image], dim=1)
                image = image.detach().cpu().numpy().transpose(1, 2, 0)
                wandb.log({"compare_output": [wandb.Image(image, caption="Comparison")]}, step=iteration)
                wandb.log({"xyz/Positions": make_3d_plot(gaussians.get_xyz)}, step=iteration)

                

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                # tb_writer.add_scalar("Loss", ema_loss_for_log, global_step=iteration)
                # tb_writer.add_scalar("NumGaussians", gaussians.get_xyz.shape[0], global_step=iteration)
                log_dict["NumGaussians"] = gaussians.get_xyz.shape[0]
                log_dict["xyz_grad_mean"] = gaussians.get_xyz.grad.mean()
                log_dict["xyz_grad_mean_magnitude"] = torch.norm(gaussians.get_xyz.grad, dim=1).mean()
                log_dict["vsp_grad_mean_magnitude"] = torch.norm(viewspace_point_tensor.grad, dim=1).mean()


                wandb.log(log_dict, step=iteration)

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration % 10_000 == 0 and iteration > 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, path)
                scene.save_model(iteration, path)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            #         wandb.log(
            #             {"NumPruned": gaussians.num_pruned,
            #              "NumSplit": gaussians.num_split,
            #              "NumCloned": gaussians.num_cloned,
            #              "NumPrunedinsplit": gaussians.num_pruned_in_split
            #             },
            #             step=iteration
            #         )
            #         log_to_wandb(gaussians, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                if iteration % batch_size == 0:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)


        if iteration < opt.densify_until_iter:
            if iteration % opt.opacity_reset_interval == 0 or (white_background and iteration == opt.densify_from_iter):
                scene.reset_col_op_decoder(gaussians.optimizer, gaussians.get_xyz)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        torch.cuda.empty_cache()

def prepare_output_and_logger():
    if os.getenv('OAR_JOB_ID'):
        unique_str=os.getenv('OAR_JOB_ID')
    else:
        unique_str = str(uuid.uuid4())
    # model_path = os.path.join("./output/", unique_str[0:10])

    model_path = os.path.join("./output/", str(datetime.now().strftime("%d/%m %H:%M:%S")))

    # Set up output folder
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer, model_path

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : SceneEG3D, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6501)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=1)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
