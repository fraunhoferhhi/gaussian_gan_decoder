import numpy as np
import torch
import wandb

from decoder_utils.camera import get_random_cam
from gaussian_renderer import render_simple
from loss_utils.lpips import perc
from loss_utils.sobel_loss import sobel_loss
from scene import CustomCam
from utils.loss_utils import l1_loss, ssim, l2_loss


def run_eval(VGG, apply_mask_to_rendering, bg, dataloader, decoder, eval_steps, gaussians, id_loss_helper, num_iter,
             use_wandb):
    # run evaluation
    lpips_list = []
    l1_list = []
    l2_list = []
    ssim_list = []
    sobel_list = []
    for _ in range(eval_steps):
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
