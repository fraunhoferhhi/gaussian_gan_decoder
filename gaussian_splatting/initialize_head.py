import sys

sys.path.append("/home/beckmann/Projects/CVG3DGaussianHeads/code/model/eg3d/eg3d/")
from gaussian_decoder.triplane_decoder import GaussianTriplaneDecoder
from eg3d_utils.plot_utils import make_3d_plot
from camera_utils import UniformCameraPoseSampler

import torch
import numpy as np



def get_random_extrinsic(horizontal_stddev=np.pi * 0.3, vertical_stddev=np.pi * 0.25):
    return UniformCameraPoseSampler.sample(
        horizontal_stddev=horizontal_stddev,
        vertical_stddev=vertical_stddev,
        radius=2.7,
        device="cuda"
    )


eg3d_model = GaussianTriplaneDecoder(num_gaussians_per_axis=120, triplane_generator_ckp="../eg3d/eg3d/networks/var3-128.pkl").to("cuda")
fov_deg = 17
fov = fov_deg / 360 * 2 * np.pi

xyz_s = []

for i in range(50):
    eg3d_model._init_xyz()

    z = torch.randn(1, 512).to(device="cuda").float()
    extrinsic = get_random_extrinsic()
    _, _ = eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

    pos = eg3d_model._xyz.detach().cpu().numpy()
    opacity = eg3d_model.gaussian_model.get_opacity.detach().cpu().numpy()
    keep_filter = (opacity > 0.1).squeeze(1)

    xyz_s.append(pos[keep_filter])


import wandb
wandb.init(project='3DGaussianHeads', dir="/home/beckmann/Projects/CVG3DGaussianHeads/code/model/gaussian-splatting/output", group="heads", name="temp")

one = torch.tensor(xyz_s[0])
two = torch.tensor(xyz_s[1])
three = np.concatenate(xyz_s)
three = torch.tensor(three)

wandb.log({"xyz/One": make_3d_plot(one)})
wandb.log({"xyz/Two": make_3d_plot(two)})




print(three.shape)
indices = torch.randperm(three.shape[0])
indices = indices[:500000]
three = three[indices]


print(three.shape)
wandb.log({"xyz/Three": make_3d_plot(three)})


