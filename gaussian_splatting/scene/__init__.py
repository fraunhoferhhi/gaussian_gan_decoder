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
import pickle
import os
import random
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2

from camera_utils import UniformCameraPoseSampler
from scene.cameras import CustomCam
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.general_utils import inverse_sigmoid


class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


class SceneEG3D:
    gaussians: GaussianModel

    def __init__(self, eg3d_model, gaussians: GaussianModel, overfit_single_id: bool = False, flame_init: bool = False):
        self.gaussians = gaussians
        self.eg3d_model = eg3d_model.to("cuda")
        self.overfit_single_id = overfit_single_id
        self.flame_init = flame_init

        self.cameras_extent = 1
        self.radius = 2.7
        self.camera_lookat_point = torch.tensor([0, 0, 0], device="cuda")
        self.z = torch.randn(1, 512).to(device="cuda").float()
        self.z_list = [self.z, torch.randn(1, 512).to(device="cuda").float()]
        self._init_xyz()

    def save(self, iteration, path):
        point_cloud_path = os.path.join(path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_model(self, iteration, path):
        snapshot_data = dict(gaussian_decoder=self.eg3d_model)
        snapshot_pkl = os.path.join(path, f'network-snapshot-{iteration:06d}.pkl')
        print("Saving snapshot to", snapshot_pkl)
        with open(snapshot_pkl, 'wb') as f:
            pickle.dump(snapshot_data, f)

    def get_random_extrinsic(self, horizontal_stddev=np.pi * 0.3, vertical_stddev=np.pi * 0.25):
        return UniformCameraPoseSampler.sample(
            horizontal_stddev=horizontal_stddev,
            vertical_stddev=vertical_stddev,
            radius=self.radius,
            device="cuda"
        )

    def get_camera_and_target(self, fov_deg=17, extrinsic=None, size=512, xyz=None, z=None):
        fov = fov_deg / 360 * 2 * np.pi

        if extrinsic is None:
            extrinsic = self.get_random_extrinsic()
        viewpoint = CustomCam(size=size, fov=fov, extr=extrinsic[0])
        if z is None:
            if not self.overfit_single_id:
                self.z = random.sample(self.z_list, 1)[0].to("cuda")
        else:
            self.z = z
        target_image, decoded_features = self.eg3d_model(self.z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)

        # self.extrinsic = extrinsic

        return viewpoint, target_image, decoded_features


    def _init_xyz(self):
        fov_deg = 17
        fov = fov_deg / 360 * 2 * np.pi
        extrinsic = self.get_random_extrinsic(0, 0)

        final_positions = []
        final_colors = []
        final_opacity = []
        num_heads = 1 if self.overfit_single_id else 1000
        with torch.no_grad():
            for _ in range(1):
                self.eg3d_model._init_xyz()
                z = self.z if self.overfit_single_id else torch.randn(1, 512).to(device="cuda").float()

                target_image, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

                if self.flame_init:
                    xyz = self.get_flame_verts(target_image)
                    self.eg3d_model._xyz = torch.nn.Parameter(xyz)
                    self.eg3d_model._xyz.requires_grad_(True)
                    _, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

                pos = self.eg3d_model._xyz.detach().cpu().numpy()
                colors = self.eg3d_model.gaussian_model._features_dc.detach().cpu().numpy()[:, 0, :]
                opacity = self.eg3d_model.gaussian_model._opacity.detach().cpu().numpy()

                opac_th = -1*np.inf if self.flame_init else 0.1
                keep_filter = (opacity > opac_th).squeeze(1)
                final_positions.append(pos[keep_filter])
                final_colors.append(colors[keep_filter])
                final_opacity.append(opacity[keep_filter])

            final_positions = np.concatenate(final_positions)
            final_colors = np.concatenate(final_colors)
            final_opacity = np.concatenate(final_opacity)

        self.gaussians.create_from_pos_col(positions=final_positions, colors=final_colors, opacity=opacity)




    # init utils
    def compute_radial_gradients(self, cube):
        cube_size = cube.shape[0]
        center = np.array([cube_size // 2, cube_size // 2, cube_size // 2])
        
        # Compute the gradient using np.gradient
        gradients = np.array(np.gradient(cube))
        
        # Initialize an empty array for radial gradient magnitudes
        radial_grad_magnitude = np.zeros(cube.shape)
        
        for x in range(cube_size):
            for y in range(cube_size):
                for z in range(cube_size):
                    # Compute the direction vector from the center to the current point
                    direction_vector = np.array([x, y, z]) - center
                    direction_vector_norm = np.linalg.norm(direction_vector)
                    if direction_vector_norm > 0:  # Avoid division by zero
                        direction_vector_normalized = direction_vector / direction_vector_norm
                        
                        # Compute the gradient vector at the current point
                        gradient_vector = gradients[:, x, y, z]
                        
                        # Project the gradient vector onto the direction vector
                        radial_gradient = np.dot(gradient_vector, direction_vector_normalized)
                        
                        # Store the magnitude of the radial gradient (projection result)
                        radial_grad_magnitude[x, y, z] = abs(radial_gradient)  # Use abs to consider magnitude only
        
        return radial_grad_magnitude

    # resetting

    def reset_col_op_decoder(self, gaussian_optimizer, xyz=None):
        optim = torch.optim.Adam(self.eg3d_model.color_opacity_decoder.parameters(), lr=0.005)

        progress_bar = tqdm(range(5000), position=0, leave=True, disable=False)
        for _ in progress_bar:
            fov_deg = np.random.rand() * 5 + 13
            fov = fov_deg / 360 * 2 * np.pi
            extrinsic = self.get_random_extrinsic()
            z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

            decoded_col_op = self.eg3d_model.forward_col_op(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)

            target_col_op = decoded_col_op.clone().detach()
            target_col_op[:, 0] = torch.min(target_col_op[:, 0], inverse_sigmoid(torch.ones_like(target_col_op[:, 0])*0.01))

            loss = (target_col_op - decoded_col_op).square().mean()
            progress_bar.set_description(f"Resetting opacity: MSE = {loss*100:0.2f}")
            loss.backward()

            optim.step()
            optim.zero_grad()

        # reset optimizer
        for param_group in gaussian_optimizer.param_groups:
            # Check if this is the opacity_decoder group
            if 'name' in param_group and param_group['name'] == 'color_opacity_decoder':
                for param in param_group['params']:
                    # Check if this param is in the optimizer state (it should be if it has gradients)
                    if param in gaussian_optimizer.state:
                        # Reset first and second moments
                        gaussian_optimizer.state[param]['exp_avg'].zero_()
                        gaussian_optimizer.state[param]['exp_avg_sq'].zero_()
        print("Done resetting {}. Continue normal training".format("opacity"))

        torch.cuda.empty_cache()

        return gaussian_optimizer

    def init_rot_scale_decoder(self, gaussian_optimizer, xyz=None):
        optim = torch.optim.Adam(self.eg3d_model.scaling_rotation_decoder.parameters(), lr=0.002)

        progress_bar = tqdm(range(500), position=0, leave=True, disable=False)
        for _ in progress_bar:
            fov_deg = np.random.rand() * 5 + 13
            fov = fov_deg / 360 * 2 * np.pi
            extrinsic = self.get_random_extrinsic()
            z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

            decoded_rot_scale = self.eg3d_model.forward_rot_scale(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)
            target_rot_scale = torch.concatenate((self.gaussians.target_rots, self.gaussians.target_scales), dim=1).detach()

            loss = (target_rot_scale - decoded_rot_scale).square().mean()
            progress_bar.set_description(f"Init rotscale: MSE = {loss:0.2f}")
            loss.backward()

            optim.step()
            optim.zero_grad()

        # reset optimizer
        for param_group in gaussian_optimizer.param_groups:
            # Check if this is the opacity_decoder group
            if 'name' in param_group and param_group['name'] == 'scaling_rotation_deocder':
                for param in param_group['params']:
                    # Check if this param is in the optimizer state (it should be if it has gradients)
                    if param in gaussian_optimizer.state:
                        # Reset first and second moments
                        gaussian_optimizer.state[param]['exp_avg'].zero_()
                        gaussian_optimizer.state[param]['exp_avg_sq'].zero_()
        print("Done resetting {}. Continue normal training".format("opacity"))

        torch.cuda.empty_cache()

        return gaussian_optimizer
