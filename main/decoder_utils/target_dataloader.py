import numpy as np
import torch

from main.camera_utils import UniformCameraPoseSampler, FOV_to_intrinsics, LookAtPoseSampler
from main.dnnlib import EasyDict
from main.marching_cube.sample import create_samples
from skimage import measure
import trimesh

from main.torch_utils import persistence


@persistence.persistent_class
class TargetDataloader:
    def __init__(
        self,
        G,
        cam_radius=2.7,
        repeat_id=1,
        truncation=1.0,
        truncation_ramp=10000,
        init_truncation=0.5,
        camera_sampling="uniform",
        vertical_stddev=0.3,
        horizontal_stddev=1.0,
        fov_offset=5,
        fov_offset_scale=12,
        use_marching_cubes=True,
        surface_thickness=0.1,
        sample_from_cube=True,  # this speeds up marching cubes significantly
        device="cuda",
        shape_res=128,
    ):
        # generator
        self.G = G
        self.device = device

        # camera
        self.vertical_stddev = np.pi * vertical_stddev
        self.horizontal_stddev = np.pi * horizontal_stddev
        self.cam_radius = cam_radius
        self.camera_sampling = camera_sampling

        # random faces / overfit one face
        self.z = torch.randn([1, self.G.z_dim], device=self.device)
        self.random_z = True
        self.internal_counter = 0
        self.same_face_counter = repeat_id
        self.truncation = truncation
        self.truncation_ramp = truncation_ramp
        self.init_truncation = init_truncation
        self.use_marching_cubes = use_marching_cubes
        self.sample_from_cube = sample_from_cube
        self.shape_res = shape_res
        self.surface_thickness = surface_thickness
        self.fov_offset = fov_offset
        self.fov_offset_scale = fov_offset_scale

    def get_data(self, z=None, camera_params=None, iteration=None, only_gan=False):
        result = EasyDict()

        self.internal_counter += 1
        if self.random_z and self.internal_counter % self.same_face_counter == 0:
            self.z = torch.randn([1, self.G.z_dim], device=self.device)

        if z is not None:
            self.z = z

        # cam params
        if camera_params is None:
            fov_deg = np.random.uniform() * self.fov_offset_scale + self.fov_offset
            intrinsics, cam2world_pose, h, v = self.get_cam(fov_deg)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            result.cam2world_pose = cam2world_pose
            result.fov_deg = fov_deg
            result.cam_h = h
            result.cam_v = v

        # run synthesis network
        with torch.no_grad():
            if iteration is not None:
                if self.truncation_ramp == 0:
                    mag = 1
                else:
                    mag = np.clip(iteration / self.truncation_ramp, 0, 1)
                truncation = self.init_truncation * (1 - mag) + self.truncation * mag
            else:
                truncation = self.truncation
            result.truncation = truncation
            ws = self.G.mapping(self.z, torch.zeros_like(camera_params), truncation_psi=truncation)
            synth = self.G.synthesis(ws, camera_params, noise_mode="const", use_cached_backbone=not self.random_z)

        if only_gan:
            return synth["image"]

        img, sigmas, feature_planes, img_mask, samples = self.pano_get_target_sigma_color(synth, ws)

        if self.use_marching_cubes:
            vertices, faces = self.marching_cubes(sigmas)
            vertices = vertices.to(self.device).float()
            vertices /= sigmas.shape[0]
            vertices -= 0.5

            face_coords = vertices[faces]

            random_positions = vertices[:0, ...]
            num_points_target = 500_000
            while random_positions.shape[0] < num_points_target:
                random_pos = torch.rand(size=[face_coords.shape[0], 3], device=self.device)
                random_pos = random_pos / torch.sum(random_pos, dim=1, keepdim=True)
                random_face_coords = torch.sum(face_coords * random_pos[..., None], dim=1)
                random_positions = torch.concat([random_positions, random_face_coords], dim=0)

            random_positions = random_positions[:num_points_target]
            random_scale = torch.clip(
                torch.randn(size=[random_positions.shape[0], 1], device=self.device) * self.surface_thickness + 1, 0, 1
            )
            random_positions = random_positions * random_scale.tile([1, 3])
        else:
            keep_mask = sigmas > 10
            random_positions = samples.reshape(128, 128, 128, 3)[keep_mask]
            random_positions = random_positions.reshape(-1, 3)

        result.img = img
        result.img_mask = img_mask
        result.vertices = random_positions
        result.gan_camera_params = camera_params
        result.sigmas = sigmas
        result.feature_planes = feature_planes
        result.ws = ws
        result.z = self.z
        return result

    def pano_get_target_sigma_color(self, synth, ws):
        img = synth["image"]
        img = torch.clip((img + 1) / 2, 0, 1)
        feature_planes = synth["feature_planes"]
        img_mask = synth.get("image_mask")
        max_batch = 1000000

        if self.sample_from_cube:
            samples, num_samples = create_samples(
                samples_per_axis=self.shape_res, voxel_origin=[0, 0, 0], cube_length=self.G.rendering_kwargs["box_warp"]
            )
            samples = samples.to(self.device)

        else:
            num_samples = self.shape_res * self.shape_res * self.shape_res
            samples = torch.rand([1, num_samples, 3], device="cuda") - 0.5

        sigmas = torch.zeros((1, num_samples, 1), device=self.device)
        transformed_ray_directions_expanded = torch.zeros((1, max_batch, 3), device=self.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with torch.no_grad():
            while head < num_samples:
                g_sample = self.G.sample_mixed(
                    samples[:, head : head + max_batch],
                    directions=transformed_ray_directions_expanded[:, : num_samples - head],
                    ws=ws,
                    noise_mode="const",
                )
                sigma = g_sample["sigma"]
                sigmas[:, head : head + max_batch] = sigma
                head += max_batch

        sigmas = sigmas.reshape((self.shape_res, self.shape_res, self.shape_res)).cpu().numpy()
        return img, sigmas, feature_planes, img_mask, samples

    def marching_cubes(self, sigmas):
        verts, faces, _, _ = measure.marching_cubes(sigmas, level=10)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        vertices = torch.tensor(mesh.vertices)
        faces = torch.tensor(mesh.faces)
        return vertices, faces

    def get_cam(self, fov_deg):
        if self.camera_sampling == "uniform":
            intrinsics = FOV_to_intrinsics(fov_deg).to(self.device)
            cam2world_pose, h, v = UniformCameraPoseSampler.sample(
                np.pi / 2,
                np.pi / 2,
                horizontal_stddev=self.horizontal_stddev,
                vertical_stddev=self.vertical_stddev,
                radius=self.cam_radius,
                device=self.device,
                return_hv=True,
            )
            return intrinsics, cam2world_pose, h, v
        elif self.camera_sampling == "normal":
            intrinsics = FOV_to_intrinsics(fov_deg).to(self.device)
            cam2world_pose, h, v = LookAtPoseSampler.sample(
                np.pi / 2,
                np.pi / 2,
                horizontal_stddev=self.horizontal_stddev,
                vertical_stddev=self.vertical_stddev,
                radius=self.cam_radius,
                device=self.device,
                return_hv=True,
            )
            return intrinsics, cam2world_pose, h, v
        else:
            raise NotImplementedError
