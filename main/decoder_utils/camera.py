import numpy as np
import torch
from camera_utils import FOV_to_intrinsics, UniformCameraPoseSampler, LookAtPoseSampler


def get_random_cam(fov_deg=10, horizontal_stddev=1.0, vertical_stddev=0.3, camera_sampling="uniform", device="cuda", return_all=False):
    cam_radius = 2.7 # do not change this, otherwise eg3d/panohead rendering is broken
    if camera_sampling == "uniform":
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        cam2world_pose = UniformCameraPoseSampler.sample(
            np.pi / 2,
            np.pi / 2,
            horizontal_stddev=horizontal_stddev,
            vertical_stddev=vertical_stddev,
            radius=cam_radius,
            device=device,
        )
    elif camera_sampling == "normal":
        intrinsics = FOV_to_intrinsics(fov_deg, device=device)
        cam2world_pose = LookAtPoseSampler.sample(
            np.pi / 2,
            np.pi / 2,
            horizontal_stddev=horizontal_stddev,
            vertical_stddev=vertical_stddev,
            radius=cam_radius,
            device=device,
        )
    else:
        raise NotImplementedError

    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    if return_all:
        return camera_params, cam2world_pose, intrinsics
    else:
        return camera_params
