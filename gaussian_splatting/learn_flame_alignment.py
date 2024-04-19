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
import wandb
import random

from faceex_arian import MasksExtractor
from simple_knn._C import distCUDA2

def main():
    wandb.init(project='3DGaussianHeads', dir="/home/beckmann/Projects/CVG3DGaussianHeads/results/learn_flame_alignment", group="learn-flame", name="debug")


    seed_value = 303
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch CUDA (for GPU computations)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch CuDNN optimizer
    torch.backends.cudnn.benchmark = False

    mex = MasksExtractor("./")
    mask_net = mex.mask_net

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(3)
    eg3d_model = GaussianTriplaneDecoder(num_gaussians_per_axis=65, triplane_generator_ckp="../eg3d/eg3d/networks/var3-128.pkl", pre_offset=False)
    scene = SceneAlignment(eg3d_model=eg3d_model, gaussians=gaussians, overfit_single_id=False, flame_init=True)
    print(gaussians._xyz.shape)
    # define learnable params here
    scale = torch.nn.Parameter(torch.tensor(2.5).float().cuda(), requires_grad=True)
    x_shift = torch.nn.Parameter(torch.tensor(0).float().cuda(), requires_grad=True)
    y_shift = torch.nn.Parameter(torch.tensor(0.).float().cuda(), requires_grad=True)
    z_shift = torch.nn.Parameter(torch.tensor(0.1).float().cuda(), requires_grad=True)

    optim = torch.optim.Adam([scale, x_shift, y_shift, z_shift], lr=0.001)

    loss = 0
    ema_loss_for_log = 0.0

    pb = tqdm(range(10000), desc='Processing', leave=True)
    for i in pb:
        with torch.no_grad():
            fov_deg = np.random.rand() * 5 + 13
            viewpoint_cam, gt_image, _ = scene.get_camera_and_target(fov_deg)

            out_image = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            out_image = (cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)*255).astype(int)

            # get mask here
            batch = np.stack([out_image]).astype(np.uint8)
            batch = mex._prepare_batch(batch)
            masks = mask_net(batch)[0].cpu().numpy().argmax(1)
            masks = masks.astype(np.uint8)[0]
            masks = np.logical_and(masks<14, masks>0).astype(np.uint8)
            mask_tensor = torch.tensor(masks, dtype=torch.bool, device="cuda").unsqueeze(0)

        # get flame stuff
            flame_xyz = scene.get_flame_verts(gt_image)
        adjusted_flame_xyz = flame_xyz.clone()
        adjusted_flame_xyz *= scale
        adjusted_flame_xyz[:, 0] += x_shift
        adjusted_flame_xyz[:, 1] += y_shift
        adjusted_flame_xyz[:, 2] += z_shift

        shift_scale_dict = {
            "scale": scale,
            "x": x_shift,
            "y": y_shift,
            "z": z_shift
        }

        scene.eg3d_model._xyz = torch.nn.Parameter(flame_xyz)
        fov = 17 / 360 * 2 * np.pi
        _, decoded_features = scene.eg3d_model(scene.z, w=None, extrinsic_eg3d=scene.extrinsic, extrinsic_gaus=scene.extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, shift_scale_dict=shift_scale_dict)

        gaussians.set_color(torch.relu(decoded_features["_features_dc"][0].unsqueeze(0).permute(1, 0 ,2)))

        positions = adjusted_flame_xyz.detach().cpu().numpy()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(positions)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        gaussians._scaling = torch.nn.Parameter(scales.requires_grad_(True))

        rots = torch.zeros((adjusted_flame_xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        gaussians._rotation = torch.nn.Parameter(rots.requires_grad_(True))

        gaussians._opacity = decoded_features["_opacity"][0]*100

        gaussians._xyz = torch.nn.Parameter(torch.tensor(np.asarray(positions)).float().cuda())

        render_pkg = render_simple(viewpoint_cam, gaussians, bg_color=background, xyz_offset=None)
        image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = image[:3]

        gt_image = gt_image.cuda()
        Ll1 = l1_loss(image*mask_tensor, gt_image*mask_tensor)
        Lssim = ssim(image*mask_tensor, gt_image*mask_tensor)
        loss += (1.0 - 0.8) * Ll1 + 0.8 * (1.0 - Lssim)
        
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        pb.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})


        if i % 4 == 0 and i > 0:
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss = 0.

            image = torch.concat([gt_image, image], dim=1)
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
            wandb.log({"compare_output": [wandb.Image(image, caption="Comparison")]}, step=i)

        log_dict = {
            "Scale": scale.item(),
            "x_shift": x_shift.item(),
            "y_shift": y_shift.item(),
            "z_shift": z_shift.item()
        }
        wandb.log(log_dict, step=i)


if __name__ == "__main__":
    main()
