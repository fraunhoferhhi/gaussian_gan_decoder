import numpy as np
import torch
import pickle

from gaussian_renderer import render_simple
from scene import GaussianModel, CustomCam


def render_with_decoder(decoder, result):
    gaussian_attr = decoder(result.z, result.gan_camera_params, result.vertices, truncation_psi=1.0)
    gaussians = GaussianModel(0)
    gaussians._xyz = gaussian_attr.xyz
    gaussians._scaling = gaussian_attr.scale
    gaussians._rotation = gaussian_attr.rotation
    gaussians._opacity = gaussian_attr.opacity
    gaussians._features_dc = gaussian_attr.color.unsqueeze(1)

    bg = torch.ones(3, device="cuda:0")
    fov = result.fov_deg / 360 * 2 * np.pi
    viewpoint = CustomCam(size=512, fov=fov, extr=result.cam2world_pose[0])
    render_obj = render_simple(viewpoint, gaussians, bg_color=bg)
    return render_obj["render"]


def load_decoder(decoder_path="./results/pano2gaussian/runbaseline_422/decoder_555000.pkl"):
    print("loading from", decoder_path)
    with open(decoder_path, "rb") as input_file:
        load_file = pickle.load(input_file)
        dataloader = load_file["dataloader"]
        decoder = load_file["decoder"]
    return decoder, dataloader


if __name__ == "__main__":
    decoder, dataloader = load_decoder()

    for i in range(10):
        result = dataloader.get_data(z=None,  camera_params=None)
        rendering = render_with_decoder(decoder, result)
        print()
