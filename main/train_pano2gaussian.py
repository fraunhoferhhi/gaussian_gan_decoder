import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from main.decoder_utils.decoder_models import ColorDecoder, GeometryDecoder
from main.decoder_utils.load_network import load_from_pkl_new_G
from main.decoder_utils.pos_encoding import Embedder
from main.decoder_utils.post_network import PostNetwork
from utils.graphics_utils import fov2focal
from utils.loss_utils import l1_loss, ssim, l2_loss
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')


sys.path.append("./PanoHead")
sys.path.append("./gaussian_splatting")
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import CustomCam
from gaussian_splatting.gaussian_renderer import render_simple
from main.decoder_utils.target_dataloader import TargetDataloader
from main.decoder_utils.seed import set_seeds
from torch.utils.tensorboard import SummaryWriter
from training.volumetric_rendering.renderer import sample_from_planes


def main():
    # manually set args
    num_iter = 100000
    seed = 3
    set_seeds(seed)

    # load / save paths
    network_pkl = "../PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl"
    outdir = f"./results/pano2gaussian/run{len(os.listdir('./results/pano2gaussian'))}"
    os.makedirs(outdir, exist_ok=True)
    writer = SummaryWriter(log_dir=outdir)

    # load / setup models
    G = load_from_pkl_new_G(network_pkl, "cuda:0")
    post_net_geom = PostNetwork().to("cuda:0")
    post_net_colo = PostNetwork().to("cuda:0")
    dataloader = TargetDataloader(G=G)
    gaussians = GaussianModel(0)
    # embedder = Embedder(include_input=True, input_dims=32, num_freqs=5)

    color_decoder = ColorDecoder(n_features=32, hidden_dim=64).to("cuda:0")
    geometry_decoder = GeometryDecoder(n_features=32, hidden_dim=64).to("cuda:0")

    optim = torch.optim.Adam(
        [
            {"params": color_decoder.parameters(), "lr": 0.005},
            {"params": post_net_colo.parameters(), "lr": 0.005},
            {"params": geometry_decoder.parameters(), "lr": 0.005},
            {"params": post_net_geom.parameters(), "lr": 0.005},
        ]
    )
    for i in tqdm(range(num_iter)):
        result = dataloader.get_data()

        post_color_planes = []
        post_geom_planes = []
        for j in range(3):
            post_color_planes.append(post_net_colo(result.feature_planes[:, j], ws=result.ws))
            post_geom_planes.append(post_net_geom(result.feature_planes[:, j], ws=result.ws))

        post_color_planes = torch.stack(post_color_planes, dim=1)
        post_geom_planes = torch.stack(post_geom_planes, dim=1)
        plane_features_color = sample_from_planes(
            G.renderer.plane_axes,
            post_color_planes,
            result.vertices.unsqueeze(0),
            padding_mode="zeros",
            box_warp=G.rendering_kwargs["box_warp"],
            triplane_depth=G.rendering_kwargs["triplane_depth"],
        )
        plane_features_geom = sample_from_planes(
            G.renderer.plane_axes,
            post_geom_planes,
            result.vertices.unsqueeze(0),
            padding_mode="zeros",
            box_warp=G.rendering_kwargs["box_warp"],
            triplane_depth=G.rendering_kwargs["triplane_depth"],
        )

        # decode values
        color, opacity = color_decoder(plane_features_color[0], result.ws)
        rotation, scale, xyz_offset = geometry_decoder(plane_features_geom[0], result.ws)

        # set attributes
        gaussians._xyz = result.vertices + xyz_offset * 0.1
        gaussians._features_dc = color.unsqueeze(1)
        gaussians._opacity = opacity
        gaussians._scaling = scale
        gaussians._rotation = rotation

        # render gaussian
        fov = result.fov_deg / 360 * 2 * np.pi
        viewpoint = CustomCam(size=512, fov=fov, extr=result.cam2world_pose[0])
        bg = torch.rand(3, device="cuda:0")
        render_obj = render_simple(viewpoint, gaussians, bg_color=bg)
        image = render_obj["render"]
        depth = render_obj["depth"]
        alpha = render_obj["alpha"]

        # calc loss
        rescale_mask = torch.nn.functional.interpolate(result.img_mask, scale_factor=(8, 8), mode="bilinear")[0]
        image = image * alpha
        target = result.img[0] * rescale_mask
        Ll2 = l2_loss(image, target)
        Lssim, ssim_map = ssim(image, target)
        offset_loss = torch.square(xyz_offset).mean()
        scale_loss = torch.prod(gaussians.scaling_activation(scale), dim=-1).mean()

        color_loss = Ll2
        geom_loss = (1.0 - Lssim) + offset_loss + scale_loss
        loss = geom_loss + color_loss

        if i % 10 == 0:
            writer.add_scalar("loss/color_loss", color_loss, i)
            writer.add_scalar("loss/geom_loss", geom_loss, i)

        if i > 500 and False:
            # estimate position for new gaussian splats
            ssim_mask = torch.any((1 - ssim_map) > 0.9, dim=0, keepdim=True)
            keep_mask = torch.logical_and(depth > 0, ssim_mask)

            xy_coords = torch.meshgrid((torch.arange(0, 512)-256), (torch.arange(0, 512)-256))
            img_coordinates = torch.stack([xy_coords[1], xy_coords[0]]).to("cuda:0") / 256

            # image space
            xyh_img_space = torch.concat([img_coordinates, torch.ones_like(depth)], dim=0)

            # cam space
            focal = fov2focal(fov, pixels=1)
            intrinsic = torch.eye(3, dtype=torch.float, device="cuda")
            intrinsic[0, 0] = focal
            intrinsic[1, 1] = focal
            xyh_cam_space = torch.inverse(intrinsic) @ xyh_img_space.reshape(3, -1)
            xyh_cam_space = xyh_cam_space[:2] / xyh_cam_space[2]

            # world space
            xyzh_img_space = torch.concat([xyh_cam_space, depth.reshape(1, -1), torch.ones_like(depth).reshape(1, -1)], dim=0)
            xyzh_world_space = torch.inverse(viewpoint.world_view_transform.T) @ xyzh_img_space
            xyz_world_space = xyzh_world_space[:3] / xyzh_world_space[3]

            # filter
            xyz_world_space = xyz_world_space[torch.tile(keep_mask, [3, 1, 1]).reshape(3, -1)].reshape(3, -1)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xyz_np = xyz_world_space.detach().cpu().numpy()
            ax.scatter(xs=xyz_np[0, ::50], ys=xyz_np[1, ::50], zs=xyz_np[2, ::50], marker="o", c="b")
            real_xyz = gaussians._xyz.detach().cpu().numpy()
            ax.scatter(xs=real_xyz[::100, 0], ys=real_xyz[::100, 1], zs=real_xyz[::100, 2], marker="o", c="r")
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            plt.show()

        # log & save checkpoints
        if i % 100 == 0:
            writer.add_image(
                "Render",
                torch.concat(
                    [
                        image,
                        target,
                        torch.abs(image - target),
                        1 - ssim_map,
                        torch.tile(rescale_mask, [3, 1, 1]),
                        torch.tile(depth, [3, 1, 1]) / depth.max(),
                    ],
                    dim=1,
                ),
                global_step=i,
            )

        if i % 1000 == 0:
            gaussians.save_ply(outdir + f"/checkpoint{i}.ply")

        loss.backward()
        optim.step()
        optim.zero_grad()



if __name__ == "__main__":
    main()
