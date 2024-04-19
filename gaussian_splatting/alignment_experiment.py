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

import os
import sys

import numpy as np
import torch
import cv2

sys.path.append("./")
sys.path.append("../eg3d/eg3d/")
from gaussian_decoder.triplane_decoder import GaussianTriplaneDecoder
from utils.loss_utils import l1_loss, ssim
from eg3d_utils.plot_utils import make_3d_plot, log_to_wandb
from gaussian_renderer import render, network_gui, render_simple
import sys
from scene import GaussianModel, SceneAlignment
from scene.cameras import CustomCam
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import random

from faceex_arian import MasksExtractor

def main():

    mex = MasksExtractor("./")
    mask_net = mex.mask_net

    seed_value = 303
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch CUDA (for GPU computations)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch CuDNN optimizer
    torch.backends.cudnn.benchmark = False

    results_path = "/home/beckmann/Projects/CVG3DGaussianHeads/results/flame_alignment"

    with torch.no_grad():
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        fov = 17 / 360 * 2 * np.pi

        gaussians = GaussianModel(3)
        eg3d_model = GaussianTriplaneDecoder(num_gaussians_per_axis=65, triplane_generator_ckp="../eg3d/eg3d/networks/var3-128.pkl", pre_offset=False)
        scene = SceneAlignment(eg3d_model=eg3d_model, gaussians=gaussians, overfit_single_id=True, flame_init=True)
        
        target_image = scene.target_image
        out_path = os.path.join(results_path, "{}.png".format("target"))
        out_image = target_image.permute(1, 2, 0).detach().cpu().numpy()
        out_image = (cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)*255).astype(int)
        cv2.imwrite(out_path, out_image)

        # get mask here
        batch = np.stack([cv2.imread(out_path)])
        batch = mex._prepare_batch(batch)
        masks = mask_net(batch)[0].cpu().numpy().argmax(1)
        masks = masks.astype(np.uint8)[0]
        masks = np.logical_and(masks<14, masks>0).astype(np.uint8)
        mask_tensor = torch.tensor(masks, dtype=torch.bool, device="cuda").unsqueeze(0)

        # add 4 more cameras / masks
        target_list = [target_image]
        mask_list = [mask_tensor]
        camera_list = [scene.extrinsic]
        fov_list = [17]

        for n in range(4):
            fov_deg = np.random.rand() * 5 + 13
            extrinsic = scene.get_random_extrinsic()
            _, gt_image, _ = scene.get_camera_and_target(fov_deg, extrinsic=extrinsic)

            out_path = os.path.join(results_path, "target_{}.png".format(n))
            out_image = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            out_image = (cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)*255).astype(int)
            cv2.imwrite(out_path, out_image)

            # get mask here
            batch = np.stack([cv2.imread(out_path)])
            batch = mex._prepare_batch(batch)
            masks = mask_net(batch)[0].cpu().numpy().argmax(1)
            masks = masks.astype(np.uint8)[0]
            masks = np.logical_and(masks<14, masks>0).astype(np.uint8)
            mask_tensor = torch.tensor(masks, dtype=torch.bool, device="cuda").unsqueeze(0)

            target_list.append(gt_image)
            mask_list.append(mask_tensor)
            camera_list.append(extrinsic)
            fov_list.append(fov_deg)


        scores = []
        scales = np.linspace(1, 3, 5)
        for scale in scales:
            shifts = np.linspace(-0.1, 0.1, 5)
            for i, x_shift in enumerate(shifts):
                for j, y_shift in enumerate(shifts):
                    for k, z_shift in enumerate(shifts):

                        scene.shift_and_set(x_shift, y_shift, z_shift, scale)

                        loss = 0.
                        for gt_image, mask, camera, fov_deg in zip(target_list, mask_list, camera_list, fov_list):
                            fov = fov_deg / 360 * 2 * np.pi

                            viewpoint_cam = CustomCam(size=512, fov=fov, extr=camera[0])

                            render_pkg = render_simple(viewpoint_cam, gaussians, bg_color=background, xyz_offset=None)
                            image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                            image = image[:3]

                            gt_image = gt_image.cuda()
                            Ll1 = l1_loss(image*mask, gt_image*mask)
                            Lssim = ssim(image*mask, gt_image*mask)
                            loss += (1.0 - 0.8) * Ll1 + 0.8 * (1.0 - Lssim)

                        out_path = os.path.join(results_path, "{}_{}_{}_{}.png".format(scale, i, j, k))
                        out_image = image.permute(1, 2, 0).detach().cpu().numpy()
                        cv2.imwrite(out_path, (cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)*255).astype(int))

                        
                        scores.append(((scale, i , j ,k), loss, Ll1, Lssim))

        scores.sort(key = lambda x: x[1])
        print(scores[:10])






if __name__ == "__main__":
    main()
