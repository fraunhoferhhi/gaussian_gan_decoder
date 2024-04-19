import sys
import click
import os

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from skimage import measure
import trimesh

import wandb

sys.path.append("./PanoHead")
sys.path.append("./gaussian_splatting")
from PanoHead import dnnlib
from PanoHead import legacy
from PanoHead.camera_utils import LookAtPoseSampler, UniformCameraPoseSampler
from PanoHead.torch_utils import misc
from PanoHead.training.triplane import TriPlaneGenerator
from PanoHead.training.volumetric_rendering.renderer import sample_from_planes, sample_from_3dgrid

from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import CustomCam
from gaussian_splatting.gaussian_renderer import render_simple
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.general_utils import inverse_sigmoid
from simple_knn._C import distCUDA2

from cvg_utils.decoders import HUGSDecoder

@click.command()
@click.option('--root-dir', 'root_dir', help='path to projects topmost dir', default="/home/beckmann/Projects/CVGGaussianGANDecoder/", required=True)
@click.option('--seed', type=int, help='random seed', default=505, required=True)
@click.option('--iters', type=int, help='how many iterations to run training', default=100000, required=True)

#TODO: add options to also train more decoders and stuff

def main(root_dir: str,
         seed: int,
         iters: int):
    set_seeds(seed)

    # manually set args
    network_pkl = os.path.join(root_dir, "PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl")
    outdir = os.path.join(root_dir, "results/pano2gaussian")
    pose_cond = 90
    pose_cond_rad = pose_cond/180*np.pi
    fov_deg = 18.837
    truncation_psi = 0.7
    truncation_cutoff = 14
    shape_res = 512
    lambda_dssim = 0.2

    # wandb
    wandb.init(project='Pano2Gaussian', dir=outdir, group="ColorDecoder", name="init-test")

    # init ml models
    G = load_G(network_pkl)
    color_decoder, optim = build_decoders_and_optim()
    gaussians = GaussianModel(3)

    pbar = tqdm(range(iters), desc="Progress")
    for iter in pbar:
        fov_deg = np.random.rand() * 13 + 6

        intrinsics = FOV_to_intrinsics(fov_deg)

        gt_image, sigmas, feature_planes, cam2world_pose, img_mask = pano_get_target_sigma_color(G, pose_cond_rad, intrinsics, truncation_cutoff, truncation_psi, shape_res)

        set_gaussian_attributes(gaussians, G, color_decoder, sigmas, feature_planes, img_mask)

        fov = fov_deg / 360 * 2 * np.pi
        viewpoint = CustomCam(size=512, fov=fov, extr=cam2world_pose[0])
        bg = torch.rand((3), device="cuda:0")
        image = render_simple(viewpoint, gaussians, bg_color=bg)["render"]
        image = image[:3]

        gt_image = gt_image.cuda().squeeze(0)
        log_dict = {}
        Ll1 = l1_loss(image, gt_image)
        Lssim = ssim(image, gt_image)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - Lssim)    

        log_dict["Loss/L1"] = Ll1
        log_dict["Loss/SSIM"] = 1.0 - Lssim
        log_dict["Loss/total"] = loss

        # optimize
        loss.backward()
        optim.step()
        optim.zero_grad()

        log_and_update_pb(log_dict, image, gt_image, iter, pbar)


# core
def set_gaussian_attributes(gaussians, G, decoder, sigmas, planes, img_mask):
    # TODO: make this code more clean / efficient
    verts, faces, _, _ = measure.marching_cubes(sigmas, level=50)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    vertices = torch.tensor(mesh.vertices).float().cuda()
    # faces = torch.tensor(mesh.faces).float().cuda()
    vertices /= 512
    vertices -= 0.5
    vertices[:, 0] *= -1
    
    sigmas = torch.tensor(sigmas.copy()).float().cuda()
    sigmas = sample_from_3dgrid(sigmas.unsqueeze(0).unsqueeze(0), vertices.unsqueeze(0)).squeeze(0)
    sigmas = sigma2opacity(sigmas, gaussians)

    plane_features = sample_from_planes(G.renderer.plane_axes, 
                                        planes, 
                                        vertices.unsqueeze(0), 
                                        padding_mode='zeros', 
                                        box_warp=G.rendering_kwargs['box_warp'], 
                                        triplane_depth=G.rendering_kwargs['triplane_depth'])
    rgbs = decoder(plane_features, vertices).squeeze(0)

    dist2 = torch.clamp_min(distCUDA2(vertices), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    rots = torch.zeros((vertices.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    gaussians._xyz = vertices
    gaussians._features_dc = rgbs.unsqueeze(1)
    gaussians._features_rest = torch.zeros((gaussians._features_dc.shape[0], 15, 3)).float().cuda()
    gaussians._opacity = inverse_sigmoid(torch.clamp(sigmas.float().cuda(), min=0.1))
    gaussians._scaling = scales
    gaussians._rotation = rots

def pano_get_target_sigma_color(G, pose_cond_rad, intrinsics, truncation_cutoff=14, truncation_psi=0.7, shape_res=512):
    with torch.no_grad():
        cam_pivot = torch.tensor([0, 0, 0], device="cuda:0")
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device="cuda:0")
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        # z and w
        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to("cuda:0")
        
        # rand camera setting
        # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        cam2world_pose = UniformCameraPoseSampler.sample(horizontal_mean=np.pi * 0.5, vertical_mean = np.pi * 0.5, horizontal_stddev=np.pi*0.4, vertical_stddev=np.pi*0.4, radius=cam_radius, device="cuda:0")
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        
        # img = G.synthesis(ws, camera_params, ws_bcg = ws_list[idx])['image']
        synth = G.synthesis(ws, camera_params)
        img = synth['image']
        feature_planes = synth['feature_planes']
        img_mask = synth["image_mask"]
        # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        max_batch=1000000

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        samples = samples.to(z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1], disable=True) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    g_sample = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')
                    sigma = g_sample["sigma"]
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        # Trim the border of the extracted cube
        pad = int(30 * shape_res / 256)
        pad_value = -1000
        sigmas[:pad] = pad_value
        sigmas[-pad:] = pad_value
        sigmas[:, :pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value

        img = torch.clip((img + 1)/ 2, 0, 1)

    return img, sigmas, feature_planes, cam2world_pose, img_mask

def load_G(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print("Reloading Modules!")
    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    G_new.neural_rendering_resolution = G.neural_rendering_resolution
    G_new.rendering_kwargs = G.rendering_kwargs
    G = G_new

    return G


# uitl functions
# logging
def log_and_update_pb(log_dict, image, gt_image, iter, pb):
    if iter % 10 == 0:
        image = torch.concat([gt_image, image], dim=1)
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        wandb.log({"compare_output": [wandb.Image(image, caption="Comparison")]}, step=iter)

    if iter % 10 == 0 and iter > 0:
        wandb.log(log_dict, step=iter)
        # pb.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})

# seeding
def set_seeds(seed_value):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed(seed_value)  # PyTorch CUDA (for GPU computations)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # PyTorch CuDNN optimizer
    torch.backends.cudnn.benchmark = False

# camera stuff
def FOV_to_intrinsics(fov_deg):
    fov_rad = fov_deg / 360 * 2 * np.pi
    focal_length = 1 / (2 * np.tan(fov_rad / 2))
    intrinsics = torch.tensor([
        [focal_length, 0, 0.5],
        [0, focal_length, 0.5],
        [0, 0, 1]]
    ).to("cuda:0").float()

    return intrinsics

# feature stuff
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def sigma2opacity(sigma, gaussian_model):
    sigma = F.softplus(sigma - 1)
    sigma = sigma * 1.0 / 512
    alpha = 1 - torch.exp(-sigma)
    alpha = gaussian_model.inverse_opacity_activation(alpha)
    alpha[torch.isneginf(alpha)] = -100
    alpha[torch.isinf(alpha)] = 100
    return alpha

def rgb2gaussiancolor(rgb):
    return np.clip(rgb[..., :3], 0, 1)

# decoder
def build_decoders_and_optim():
    print("Setting up decoders and optim")
    decoder = HUGSDecoder(n_features=32+3, out_features=[3], hidden_dim=64).to("cuda:0")
    optim = torch.optim.Adam(decoder.parameters(), lr=0.003)

    return decoder, optim




if __name__ == '__main__':
    main()