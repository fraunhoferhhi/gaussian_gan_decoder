import sys

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from skimage import measure
import trimesh

sys.path.append("./PanoHead")
sys.path.append("./gaussian_splatting")
from PanoHead import dnnlib
from PanoHead import legacy
from PanoHead.camera_utils import LookAtPoseSampler, UniformCameraPoseSampler
from PanoHead.torch_utils import misc
from PanoHead.training.triplane import TriPlaneGenerator

from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import CustomCam
from gaussian_splatting.gaussian_renderer import render_simple


def main():
    # manually set args
    network_pkl = "/home/beckmann/Projects/PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl"
    pose_cond = 90
    fov_deg = 18.837
    truncation_psi = 0.7
    truncation_cutoff = 14
    shape_res = 512

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda:0')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    outdir = "/home/beckmann/Projects/CVGGaussianGANDecoder/results/pano2gaussian"

    pose_cond_rad = pose_cond/180*np.pi

    # intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    fov_rad = fov_deg / 360 * 2 * np.pi
    focal_length = 1 / (2 * np.tan(fov_rad / 2))
    intrinsics = torch.tensor([
        [focal_length, 0, 0.5],
        [0, focal_length, 0.5],
        [0, 0, 1]]
    ).to("cuda:0").float()

    seed_idx = 0
    seed = 3

    cam_pivot = torch.tensor([0, 0, 0], device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(pose_cond_rad, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    
    # z and w
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    
    angle_y = 0
    angle_p = -0.2

    # rand camera setting
    # cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    cam2world_pose = UniformCameraPoseSampler.sample(horizontal_mean=np.pi * 0.5, vertical_mean = np.pi * 0.5, horizontal_stddev=np.pi * 0., vertical_stddev=np.pi * 0., radius=cam_radius, device="cuda:0")
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    
    # img = G.synthesis(ws, camera_params, ws_bcg = ws_list[idx])['image']
    synth = G.synthesis(ws, camera_params)
    img = synth['image']
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_pano.png')

    max_batch=1000000

    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
    rgbs = torch.zeros((samples.shape[0], samples.shape[1], 32), device=z.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                g_sample = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')
                sigma = g_sample["sigma"]
                rgb = g_sample["rgb"]
                sigmas[:, head:head+max_batch] = sigma
                rgbs[:, head:head+max_batch] = rgb
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    rgbs = rgbs.reshape((shape_res, shape_res, shape_res, 32)).cpu().numpy()
    rgbs = np.flip(rgbs, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    # Use the marching cubes algorithm to find the surface
    verts, faces, _, _ = measure.marching_cubes(sigmas, level=10)

    # Convert the vertices and faces into a mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # Now you have the mesh, which you can manipulate further or save to a file
    # For example, saving the mesh to an OBJ file
    mesh.export('results/pano2gaussian/mesh.obj')

    # If you need the vertices and faces as numpy arrays, you can do:
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # build gaussian model
    gaussians = GaussianModel(3)

    # sample info from cubes
    xyz_sigmas = sample_from_cube(sigmas, vertices)
    xyz_sigmas = sigma2opacity(torch.tensor(xyz_sigmas), gaussians).unsqueeze(-1).detach().cpu().numpy()

    xyz_rgbs = sample_from_cube(rgbs, vertices)
    xyz_rgbs = rgb2gaussiancolor(xyz_rgbs)

    xyz_rgbs *= 0
    xyz_rgbs[:, 0] = 1

    # set gaussian attributes
    vertices /= 512
    vertices -= 0.5
    vertices[:, 0] *= -1
    gaussians.create_from_pos_col(positions=vertices, colors=xyz_rgbs, opacity=xyz_sigmas)

    # render
    fov = fov_deg / 360 * 2 * np.pi
    viewpoint = CustomCam(size=512, fov=fov, extr=cam2world_pose[0])
    bg = torch.rand((3), device="cuda:0")
    image = render_simple(viewpoint, gaussians, bg_color=bg)["render"]
    image = image[:3]

    # save image
    image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(image.cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_gaussian.png')


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

def sample_from_cube(cube, xyz):
    """
    Performs trilinear interpolation for each vertex in a given set of vertices within a 3D volume.

    :param volume: A 3D numpy array.
    :param vertices: An (N, 3) numpy array of vertices, where each row is (x, y, z).
    :return: A numpy array of interpolated values at each vertex.
    """
    interpolated_values = []
    for vertex in xyz:
        x, y, z = vertex
        
        # Floor coordinates
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        # Ceiling coordinates
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
        
        # Ensure coordinates are within the volume bounds
        x0, x1 = max(0, x0), min(cube.shape[0] - 1, x1)
        y0, y1 = max(0, y0), min(cube.shape[1] - 1, y1)
        z0, z1 = max(0, z0), min(cube.shape[2] - 1, z1)
        
        # Fractional part of the coordinates
        xd, yd, zd = x - x0, y - y0, z - z0
        
        # Interpolate along x axis (8 corners of the cube)
        c00 = cube[x0, y0, z0] * (1 - xd) + cube[x1, y0, z0] * xd
        c01 = cube[x0, y0, z1] * (1 - xd) + cube[x1, y0, z1] * xd
        c10 = cube[x0, y1, z0] * (1 - xd) + cube[x1, y1, z0] * xd
        c11 = cube[x0, y1, z1] * (1 - xd) + cube[x1, y1, z1] * xd
        
        # Interpolate along y axis
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        # Interpolate along z axis
        c = c0 * (1 - zd) + c1 * zd
        
        interpolated_values.append(c)
    
    return np.array(interpolated_values)

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

if __name__ == '__main__':
    main()