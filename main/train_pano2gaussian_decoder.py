import copy
import os
import pickle
import sys
import click
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
import wandb

sys.path.append("../")
sys.path.append("../gaussian_splatting")
matplotlib.use("Qt5Agg")

from main.decoder_utils.camera import get_random_cam
from main.loss_utils.id_loss import IDLoss
from torch.utils.tensorboard import SummaryWriter
from main.decoder_utils.load_network import load_from_pkl_new_G
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import CustomCam
from gaussian_splatting.gaussian_renderer import render_simple
from main.decoder_utils.target_dataloader import TargetDataloader
from main.decoder_utils.seed import set_seeds
from main.loss_utils.sobel_loss import sobel_loss
from main.loss_utils.lpips import perc, NvidiaVGG16
from main.load_decoder import load_decoder

@click.command()
# training settings
@click.option("--seed", help="Random seed", type=int, default=303, show_default=True)
@click.option("--lr", help="Learning rate", type=float, default=0.00009, show_default=True)
@click.option("--num_iter", help="Number of training iterations", type=int, default=100_001, show_default=True)
@click.option("--eval_steps", type=int, default=10_000, show_default=True)
# loss weights
@click.option("--l1_weight", help="Weight for l1 loss", type=float, default=0.2, show_default=True)
@click.option("--l2_weight", help="Weight for mse loss", type=float, default=0.1, show_default=True)
@click.option("--lpips_weight", help="Weight for perceptual loss", type=float, default=1.0, show_default=True)
@click.option("--ssim_weight", help="Weight for dssim loss", type=float, default=0.5, show_default=True)
@click.option("--sobel_weight", help="Weight for sobel loss", type=float, default=0.2, show_default=True)
@click.option("--id_loss_weight", help="Weight for sobel loss", type=float, default=1.0, show_default=True)
# decoder options
@click.option("--generator_arch", help="[eg3d_ffhq, eg3d_lpff, eg3d_cats, panohead]", type=str, default="panohead", show_default=True)
@click.option("--load_checkpoint", type=str, default="", show_default=True)
@click.option("--decoder_type", help="[sequential, parallel, sequential_reversed]", type=str, default="sequential_reversed", show_default=True)
@click.option("--use_pos_encoding", type=bool, default=False, show_default=True)
@click.option("--use_gen_finetune", type=bool, default=True, show_default=True)
@click.option("--triplane_sr", help="[None, 512, 1024]", type=str, default="None", show_default=True)
@click.option("--hidden_dim", type=int, default=128, show_default=True)
@click.option("--use_marching_cubes", type=bool, default=True, show_default=True)
@click.option("--sample_from_cube", type=bool, default=True, show_default=True)
@click.option("--surface_thickness", type=float, default=0.1, show_default=True)
@click.option("--apply_mask_to_rendering", type=bool, default=False, show_default=True)
# target data
@click.option("--truncation", help="interpolates the latent towards avg", type=float, default=1.0, show_default=True)
@click.option("--init_truncation", type=float, default=1.0, show_default=True)
@click.option("--truncation_ramp", help="steps", type=int, default=20000, show_default=True)
@click.option("--camera_sampling", help="Chose from [uniform, normal]", type=str, default="normal", show_default=True)
@click.option("--repeat_id", type=int, default=1, show_default=True)
# wandb params
@click.option("--run_name", help="Run name for wandb logger", type=str, default="test", show_default=True)
@click.option("--save_model_interval", type=int, default=25000, show_default=True)
@click.option("--logging_interval", help="how often to log features", type=int, default=1000, show_default=True)
@click.option("--group", help="Group for wandb logger", type=str, default="train_eg3d", show_default=True)
@click.option("--disable_tqdm", type=bool, default=False, show_default=True)
@click.option("--use_wandb", type=bool, default=True, show_default=True)
def main(
        seed,
        lr,
        num_iter,
        eval_steps,
        l1_weight,
        l2_weight,
        lpips_weight,
        ssim_weight,
        sobel_weight,
        id_loss_weight,
        generator_arch,
        load_checkpoint,
        decoder_type,
        use_pos_encoding,
        use_gen_finetune,
        triplane_sr,
        hidden_dim,
        use_marching_cubes,
        sample_from_cube,
        surface_thickness,
        apply_mask_to_rendering,
        truncation,
        init_truncation,
        truncation_ramp,
        camera_sampling,
        repeat_id,
        run_name,
        save_model_interval,
        logging_interval,
        group,
        disable_tqdm,
        use_wandb,
):
    set_seeds(seed)
    config = dict(
        seed=seed,
        lr=lr,
        num_iter=num_iter,
        l1_weight=l1_weight,
        l2_weight=l2_weight,
        lpips_weight=lpips_weight,
        ssim_weight=ssim_weight,
        sobel_weight=sobel_weight,
        id_loss_weight=id_loss_weight,
        decoder_type=decoder_type,
        use_pos_encoding=use_pos_encoding,
        use_gen_finetune=use_gen_finetune,
        triplane_sr=triplane_sr,
        hidden_dim=hidden_dim,
        truncation=truncation,
        camera_sampling=camera_sampling,
        repeat_id=repeat_id,
    )
    device = "cuda:0"
    os.makedirs("./results", exist_ok=True)
    network_pkl = ""
    if generator_arch == "panohead":
        network_pkl = "../PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl"
        os.makedirs("./results/pano2gaussian", exist_ok=True)
        number = len(os.listdir("./results/pano2gaussian"))
        outdir = f"./results/pano2gaussian/run{run_name}_{number}"
        os.makedirs(outdir, exist_ok=True)
        sys.path.append("../PanoHead")
        vertical_std = 0.3
        horizontal_std = 1.0
        fov_offset = 5
        fov_offset_scale = 12
        bg = torch.tensor([0.55717, 0.52256, 0.51045], dtype=torch.float, device=device)

    elif generator_arch.startswith("eg3d"):
        if generator_arch == "eg3d_lpff":
            network_pkl = "../eg3d/checkpoints_lpff/var3-128.pkl"
        elif generator_arch == "eg3d_ffhq":
            network_pkl = "../eg3d/checkpoints/ffhq512-128.pkl"
        elif generator_arch == "eg3d_cats":
            network_pkl = "../eg3d/checkpoints/afhqcats512-128.pkl"
            id_loss_weight = 0
        os.makedirs("./results/eg3d", exist_ok=True)
        number = len(os.listdir("./results/eg3d"))
        outdir = f"./results/eg3d/run{run_name}_{number}"
        os.makedirs(outdir, exist_ok=True)
        sys.path.append("../eg3d")
        generator_arch = "eg3d"
        vertical_std = 0.2
        horizontal_std = 0.2
        fov_offset = 5
        fov_offset_scale = 12
        bg = torch.tensor([0., 0., 0.], dtype=torch.float, device=device)

    if use_wandb:
        wandb_run_name = "{}_{}".format(run_name, number)
        wandb.init(project="GaussianHeadDecoder", dir=outdir, group=group, name=wandb_run_name)
    writer = SummaryWriter(log_dir=outdir)

    # import decoder models here since they depend on the sys path
    from main.decoder_models.sequential_decoder import SequentialDecoder
    from main.decoder_models.parallel_decoder import ParallelDecoder
    from main.decoder_models.sequential_decoder_reverse import SequentialDecoderReverse

    # load / setup models
    assert network_pkl != "", f"invalid generator_arch: {generator_arch}"
    G = load_from_pkl_new_G(network_pkl, device, generator_arch)
    G_gaussian = copy.deepcopy(G).train().requires_grad_(True).to(device)
    if load_checkpoint != "":
        decoder, dataloader = load_decoder(load_checkpoint)
    else:
        if decoder_type == "sequential":
            decoder = SequentialDecoder(
                G_gaussian,
                use_xyz_embedding=use_pos_encoding,
                hidden_dim=hidden_dim,
                use_gen_finetune=use_gen_finetune,
                triplane_sr=triplane_sr,
            )
        elif decoder_type == "sequential_reversed":
            decoder = SequentialDecoderReverse(
                G_gaussian,
                use_xyz_embedding=use_pos_encoding,
                hidden_dim=hidden_dim,
                use_gen_finetune=use_gen_finetune,
                triplane_sr=triplane_sr,
            )
        elif decoder_type == "parallel":
            decoder = ParallelDecoder(
                G_gaussian,
                use_xyz_embedding=use_pos_encoding,
                hidden_dim=hidden_dim,
                use_gen_finetune=use_gen_finetune,
                triplane_sr=triplane_sr,
            )
        else:
            raise NotImplementedError

        dataloader = TargetDataloader(
            G=G,
            repeat_id=repeat_id,
            truncation=truncation,
            init_truncation=init_truncation,
            truncation_ramp=truncation_ramp,
            camera_sampling=camera_sampling,
            sample_from_cube=sample_from_cube,
            use_marching_cubes=use_marching_cubes,
            surface_thickness=surface_thickness,
            horizontal_stddev=horizontal_std,
            vertical_stddev=vertical_std,
            fov_offset=fov_offset,
            fov_offset_scale=fov_offset_scale
        )

    VGG = NvidiaVGG16()
    id_loss_helper = IDLoss()
    optim = torch.optim.Adam([{"params": decoder.get_params_custom(), "lr": lr}])
    # setup gaussian model
    gaussians = GaussianModel(0)

    for i in tqdm(range(num_iter), disable=disable_tqdm):
        result = dataloader.get_data(iteration=i)
        decoded_attributes = decoder(result.z, result.gan_camera_params, result.vertices,
                                     truncation_psi=result.truncation)

        # set attributes
        gaussians._xyz = decoded_attributes.xyz
        gaussians._scaling = decoded_attributes.scale
        gaussians._rotation = decoded_attributes.rotation
        gaussians._opacity = decoded_attributes.opacity
        gaussians._features_dc = decoded_attributes.color.unsqueeze(1)

        # render gaussian
        fov = result.fov_deg / 360 * 2 * np.pi
        viewpoint = CustomCam(size=512, fov=fov, extr=result.cam2world_pose[0])
        render_obj = render_simple(viewpoint, gaussians, bg_color=bg)
        image = render_obj["render"]
        image = image[:3, ...]
        target = result.img[0]

        # apply mask
        if apply_mask_to_rendering:
            rescale_mask = torch.nn.functional.interpolate(result.img_mask, scale_factor=(8, 8), mode="bilinear")[0]
            image = image * rescale_mask + 1 - rescale_mask
            target = target * rescale_mask + 1 - rescale_mask

        # calc loss
        Ll1 = l1_loss(image, target)
        Ll2 = l2_loss(image, target)
        Lssim, ssim_map = ssim(image, target)
        Lssim = 1.0 - Lssim
        sobel, sobel_image = sobel_loss(image, target)
        lpips = perc(target.unsqueeze(0), image.unsqueeze(0), vgg=VGG, downsampling=True)

        # only calculate id loss for frontal images without much zoom
        id_cam_threshold = np.pi * 0.2
        frontal_angle = np.abs(result.cam_h.item() - np.pi / 2) < id_cam_threshold and np.abs(
            result.cam_v.item() - np.pi / 2) < id_cam_threshold
        good_zoom = result.fov_deg > 8
        if good_zoom and frontal_angle:
            id_loss = id_loss_helper(image.unsqueeze(0), target.unsqueeze(0))
        else:
            id_loss = 0

        loss = Ll1 * l1_weight + Ll2 * l2_weight + lpips * lpips_weight + Lssim * ssim_weight + sobel * sobel_weight + id_loss * id_loss_weight

        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % logging_interval == 0:
            compare_image = torch.concat([image, target], dim=2)
            writer.add_image(
                "Render",
                compare_image,
                global_step=i,
            )
            if use_wandb:
                wandb.log({"compare_output": [wandb.Image(compare_image, caption="compare")]}, step=i)

        # log & save checkpoints
        if i % 50 == 0:
            log_dict = {}
            log_dict["General/truncation"] = result.truncation

            log_dict["Loss/L1"] = Ll1
            log_dict["Loss/L2"] = Ll2
            log_dict["Loss/LPIPS"] = lpips
            log_dict["Loss/DSSIM"] = Lssim
            log_dict["Loss/Sobel"] = sobel
            if id_loss != 0:
                log_dict["Loss/id"] = id_loss
            log_dict["Loss/total"] = loss

            if use_wandb:
                wandb.log(log_dict, step=i)

        if i % logging_interval == 0:
            compare_image = torch.concat([image, target], dim=2)

            if use_wandb:
                wandb.log({"compare_output": [wandb.Image(compare_image, caption="Comparison")]}, step=i)

            writer.add_scalar("General/truncation", result.truncation, i)
            writer.add_scalar("loss/L1", Ll1, i)
            writer.add_scalar("loss/L2", Ll2, i)
            writer.add_scalar("loss/LPIPS", lpips, i)
            writer.add_scalar("loss/SSIM", Lssim, i)
            writer.add_scalar("loss/Sobel", sobel, i)
            if id_loss != 0:
                writer.add_scalar("loss/id", id_loss, i)
            writer.add_scalar("loss/total loss", loss, i)

        if i % save_model_interval == 0:
            gaussians.save_ply(outdir + f"/checkpoint{i}.ply")
            snapshot_data = dict(
                training_set_kwargs=config,
                decoder=decoder,
                dataloader=dataloader,
            )
            if not use_pos_encoding:
                with open(f"{outdir}/decoder_{i:06d}.pkl", "wb") as f:
                    pickle.dump(snapshot_data, f)
            torch.save(decoder.state_dict(), f"{outdir}/decoder{i}.pt")

        if i % 100_000 == 0 and i > 0:
            run_eval(VGG, apply_mask_to_rendering, bg, dataloader, decoder, eval_steps, gaussians, id_loss_helper,
                     i,
                     use_wandb)

        if i % 10 == 0:
            torch.cuda.empty_cache()

    run_eval(VGG, apply_mask_to_rendering, bg, dataloader, decoder, eval_steps, gaussians, id_loss_helper, num_iter,
             use_wandb)


def run_eval(VGG, apply_mask_to_rendering, bg, dataloader, decoder, eval_steps, gaussians, id_loss_helper, num_iter,
             use_wandb):
    # run evaluation
    lpips_list = []
    l1_list = []
    l2_list = []
    ssim_list = []
    sobel_list = []
    for i in range(eval_steps):
        with torch.no_grad():
            fov_deg = np.random.uniform() * 12 + 5
            cam, cam2world_pose, intrinsics = get_random_cam(fov_deg=fov_deg, return_all=True)
            result = dataloader.get_data(camera_params=cam)
            decoded_attributes = decoder(result.z, result.gan_camera_params, result.vertices, truncation_psi=1.0)

            # set attributes
            gaussians._xyz = decoded_attributes.xyz
            gaussians._scaling = decoded_attributes.scale
            gaussians._rotation = decoded_attributes.rotation
            gaussians._opacity = decoded_attributes.opacity
            gaussians._features_dc = decoded_attributes.color.unsqueeze(1)

            # render gaussian
            fov = fov_deg / 360 * 2 * np.pi
            viewpoint = CustomCam(size=512, fov=fov, extr=cam2world_pose[0])
            render_obj = render_simple(viewpoint, gaussians, bg_color=bg)
            image = render_obj["render"]
            target = result.img[0]

            # apply mask
            if apply_mask_to_rendering:
                rescale_mask = torch.nn.functional.interpolate(result.img_mask, scale_factor=(8, 8), mode="bilinear")[0]
                image = image * rescale_mask + 1 - rescale_mask
                target = target * rescale_mask + 1 - rescale_mask

            Ll1 = l1_loss(image, target)
            Ll2 = l2_loss(image, target)
            Lssim, ssim_map = ssim(image, target)
            sobel, sobel_image = sobel_loss(image, target)
            lpips = perc(target.unsqueeze(0), image.unsqueeze(0), vgg=VGG, downsampling=True)

            l1_list.append(Ll1.item())
            l2_list.append(Ll2.item())
            ssim_list.append(Lssim.item())
            sobel_list.append(sobel.item())
            lpips_list.append(lpips.item())
    avg_lpips_list = np.array(lpips_list).mean()
    avg_l1_list = np.array(l1_list).mean()
    avg_l2_list = np.array(l2_list).mean()
    avg_ssim_list = np.array(ssim_list).mean()
    avg_sobel_list = np.array(sobel_list).mean()
    id_sim_list = []
    for i in range(eval_steps):
        with torch.no_grad():
            fov_deg = 10
            cam, cam2world_pose, intrinsics = get_random_cam(fov_deg=fov_deg, horizontal_stddev=0.1,
                                                             vertical_stddev=0.1, camera_sampling="normal",
                                                             return_all=True)
            result = dataloader.get_data(camera_params=cam)
            decoded_attributes = decoder(result.z, result.gan_camera_params, result.vertices, truncation_psi=1.0)

            # set attributes
            gaussians._xyz = decoded_attributes.xyz
            gaussians._scaling = decoded_attributes.scale
            gaussians._rotation = decoded_attributes.rotation
            gaussians._opacity = decoded_attributes.opacity
            gaussians._features_dc = decoded_attributes.color.unsqueeze(1)

            # render gaussian
            fov = fov_deg / 360 * 2 * np.pi
            viewpoint = CustomCam(size=512, fov=fov, extr=cam2world_pose[0])
            render_obj = render_simple(viewpoint, gaussians, bg_color=bg)
            image = render_obj["render"]
            target = result.img[0]

            # apply mask
            if apply_mask_to_rendering:
                rescale_mask = torch.nn.functional.interpolate(result.img_mask, scale_factor=(8, 8), mode="bilinear")[0]
                image = image * rescale_mask + 1 - rescale_mask
                target = target * rescale_mask + 1 - rescale_mask

            id_loss = 1 - id_loss_helper(image.unsqueeze(0), target.unsqueeze(0))
            id_sim_list.append(id_loss.item())
    avg_id_sim_list = np.array(id_sim_list).mean()
    log_dict = {}
    log_dict["Metrics10k/L1"] = avg_l1_list
    log_dict["Metrics10k/L2"] = avg_l2_list
    log_dict["Metrics10k/LPIPS"] = avg_lpips_list
    log_dict["Metrics10k/DSSIM"] = avg_ssim_list
    log_dict["Metrics10k/Sobel"] = avg_sobel_list
    log_dict["Metrics10k/id"] = avg_id_sim_list
    print(log_dict)
    if use_wandb:
        wandb.log(log_dict, step=num_iter)


if __name__ == "__main__":
    main()
