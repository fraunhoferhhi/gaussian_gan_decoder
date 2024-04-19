import os
import pickle
import random

import numpy as np
import torch
from skimage.transform import estimate_transform, warp
from smplx import FLAME

from camera_utils import UniformCameraPoseSampler
from scene import CustomCam, GaussianModel


class ResnetEncoder:
    pass


class SceneEG3D_flame:
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

        if flame_init:
            # init deca decoder
            E_flame = ResnetEncoder(outsize=236)
            ckpt = torch.load("/home/beckmann/Projects/DECA/data/deca_model.tar")
            copy_state_dict(E_flame.state_dict(), ckpt["E_flame"])
            E_flame.to("cuda")
            E_flame.eval()

            self.E_flame = E_flame
            self.flamelayer = FLAME(get_config()).to("cuda")
            self.fan = FAN()

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
        target_image, decoded_features = self.eg3d_model(self.z, w=None, extrinsic_eg3d=extrinsic,
                                                         extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov,
                                                         xyz=xyz, only_render_eg3d=True)

        self.extrinsic = extrinsic

        return viewpoint, target_image, decoded_features

    def _init_xyz(self):
        fov_deg = 17
        fov = fov_deg / 360 * 2 * np.pi
        extrinsic = self.get_random_extrinsic(0, 0)

        final_positions = []
        final_colors = []
        final_opacities = []
        num_heads = 1 if self.overfit_single_id else 1000
        with torch.no_grad():
            for _ in range(1):
                self.eg3d_model._init_xyz()
                z = self.z if self.overfit_single_id else torch.randn(1, 512).to(device="cuda").float()

                target_image, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic,
                                                  fov_rad_eg3d=fov, fov_rad_gaus=fov)

                # if self.flame_init:
                #     xyz = self.get_flame_verts(target_image)
                #     self.eg3d_model._xyz = torch.nn.Parameter(xyz)
                #     self.eg3d_model._xyz.requires_grad_(True)
                #     _, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

                pos = self.eg3d_model._xyz.detach().cpu().numpy()
                colors = self.eg3d_model.gaussian_model._features_dc.detach().cpu().numpy()[:, 0, :]
                opacity = self.eg3d_model.gaussian_model.get_opacity.detach().cpu().numpy()

                keep_filter = (opacity > 0.09).squeeze(1)
                final_positions.append(pos[keep_filter])
                final_colors.append(colors[keep_filter])
                final_opacities.append(opacity[keep_filter])

            final_positions = np.concatenate(final_positions)
            final_colors = np.concatenate(final_colors)
            final_opacities = np.concatenate(final_opacities)

        self.gaussians.create_from_pos_col(positions=final_positions, colors=final_colors)

    # flame stuff
    def get_flame_verts(self, gt_image):
        image = gt_image.detach().clone()

        with torch.no_grad():
            E_flame = self.E_flame

            # make exactly like flame
            image = image.permute(1, 2, 0).detach().cpu().numpy() * 255
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            resolution_inp = 224

            det = self.fan
            bbox, bbox_type = det.run(image.astype(np.uint8))
            try:
                left = bbox[0];
                right = bbox[2]
                top = bbox[1];
                bottom = bbox[3]
            except:
                print("no face detected! run original image")
                left = 0;
                right = h - 1;
                top = 0;
                bottom = w - 1

            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * 1.25)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
            DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.

            dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
            dst_image = dst_image.transpose(2, 0, 1)

            image = torch.tensor(dst_image).float().cuda()

            flame_out = E_flame(image.unsqueeze(0))
            shape = flame_out[:, :100]
            exp = flame_out[:, 150:200]
            pose = flame_out[:, 200:206]

            pose[:, :3] = 0  # neutralize pose

            flamelayer = self.flamelayer
            faces = flamelayer.faces_tensor

            vertices, _, _ = flamelayer(shape, exp, pose)
            vertices = vertices.squeeze(0)

            xyz = torch.zeros((faces.shape[0], 3)).cuda()

            xyz = []
            normals = []
            for i in range(faces.shape[0]):
                triangle_area = self.compute_triangle_area(vertices[faces[i], :])
                n = self.triangle_area_to_n(triangle_area)
                face_points = self.generate_points_on_triangle(vertices[faces[i], :], n)
                xyz.append(face_points)
                _, normal, _ = self._get_triangle_info(vertices[faces[i], :])
                normals += [normal.unsqueeze(0)] * n

            xyz = torch.concatenate(xyz, dim=0).cuda()
            normals = torch.concatenate(normals, dim=0).cuda()

            stack = [xyz]
            for factor in np.linspace(0, 0.075, 10)[1:]:
                new = xyz + factor * normals
                stack.append(new)

            xyz = torch.concatenate(stack, dim=0)

            xyz = xyz * 2.5
            xyz[:, 1] += 0.1

            # shoulder parts
            x_shoulder = torch.rand((7500, 1)).cuda() - 0.5
            y_shoulder = torch.rand((7500, 1)).cuda() * -0.3 - 0.2
            z_shoulder = torch.rand((7500, 1)).cuda() * 0.4 - 0.25

            shoulder = torch.concatenate((x_shoulder, y_shoulder, z_shoulder), dim=-1)

            # wall parts
            x_left = torch.rand((3000, 1)).cuda() * 0.05 - 0.5
            y_left = torch.rand((3000, 1)).cuda() - 0.5
            z_left = torch.rand((3000, 1)).cuda() - 0.5

            left = torch.concatenate((x_left, y_left, z_left), dim=-1)
            right = torch.concatenate((-1 * x_left, y_left, z_left), dim=-1)

            x_back = torch.rand((3000, 1)).cuda() * 0.9 - 0.45
            y_back = torch.rand((3000, 1)).cuda() - 0.5
            back = torch.concatenate((x_back, y_back, x_left), dim=-1)

            walls = torch.concatenate((left, right, back), dim=0)

            xyz = torch.concatenate((xyz, shoulder, walls), dim=0)

        return xyz

    def _get_triangle_info(self, vertices):
        v1 = vertices[0]
        v2 = vertices[1]
        v3 = vertices[2]

        # Compute normal vector n
        n = F.normalize(torch.cross(v2 - v1, v3 - v1), dim=0)

        # Compute r2 as the normalized vector from the centroid to v1
        m = (v1 + v2 + v3) / 3.0
        r2 = F.normalize(v1 - m, dim=0)

        # Compute r3 using the Gram-Schmidt process
        r1 = n
        r3 = F.normalize(torch.cross(r1, r2), dim=0)

        # Rotation matrix R
        R = torch.stack([r1, r2, r3], dim=1)
        r = self.rotation_matrix_to_quaternion(R)

        # Scaling vector S
        s2 = torch.norm(m - v1)
        s3 = torch.dot((v2 - m), r3)
        S = torch.tensor([1e-3, s2, s3])  # Assuming s1 is a small constant for numerical stability

        return r, n, S

    def rotation_matrix_to_quaternion(self, R):
        # Make sure the input matrix is of float type for precision
        R = R.float()

        # Preallocate the quaternion tensor
        q = torch.zeros(4)

        # Compute the trace of the matrix
        trace = torch.trace(R)

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2
            q[0] = 0.25 * s
            q[1] = (R[2, 1] - R[1, 2]) / s
            q[2] = (R[0, 2] - R[2, 0]) / s
            q[3] = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

        # Normalize the quaternion to ensure it's a unit quaternion
        q = q / torch.norm(q)
        return q

    def _get_face_normals(self, vertices):
        # mean position
        T = vertices.mean(1)

        # move to origin
        A = vertices[:, 0] - T
        B = vertices[:, 1] - T
        C = vertices[:, 2] - T

        # direction vector of A and B
        direction_vec = B - A
        direction_vec = direction_vec / direction_vec.norm(p=2, dim=1, keepdim=True)

        # normal vector
        normal_vec = torch.cross(input=B - A, other=C - A)
        normal_vec = normal_vec / normal_vec.norm(p=2, dim=1, keepdim=True)

        return normal_vec

    def generate_points_on_triangle(self, vertices, n):
        """
        Generate n points on the face of a triangle using PyTorch.

        Parameters:
        - vertices: A (3, 3) tensor containing the xyz coordinates of the triangle's vertices.
        - n: The number of points to generate.

        Returns:
        - A (n, 3) tensor containing the xyz coordinates of the generated points.
        """
        points = torch.zeros((n, 3))
        for i in range(n):
            # Generate random barycentric coordinates
            s = torch.rand(2)
            s, _ = torch.sort(s)  # Ensure the generated values are in ascending order
            t1, t2 = s[0], s[1] - s[0]
            t3 = 1 - s[1]

            # Convert barycentric coordinates to Cartesian coordinates
            point = t1 * vertices[0] + t2 * vertices[1] + t3 * vertices[2]
            points[i] = point

        return points

    def compute_triangle_area(self, vertices):
        # Compute vectors of two sides of the triangle
        side1 = vertices[1] - vertices[0]
        side2 = vertices[2] - vertices[0]

        # Compute cross product of the two sides
        cross_product = torch.cross(side1, side2)

        # Compute area of the triangle using half of the magnitude of the cross product
        area = 0.5 * torch.norm(cross_product)

        return area

    def triangle_area_to_n(self, face_space, min=6.225e-9, max=0.0002):
        n = (face_space.item() - min) / (max - min)
        n *= 11
        n += 1

        return round(n)

    def reset_col_op_decoder(self, gaussian_optimizer, xyz=None):
        optim = torch.optim.Adam(self.eg3d_model.color_opacity_decoder.parameters(), lr=0.005)

        progress_bar = tqdm(range(5000), position=0, leave=True, disable=False)
        for _ in progress_bar:
            fov_deg = np.random.rand() * 5 + 13
            fov = fov_deg / 360 * 2 * np.pi
            extrinsic = self.get_random_extrinsic()
            z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

            decoded_col_op = self.eg3d_model.forward_col_op(z, w=None, extrinsic_eg3d=extrinsic,
                                                            extrinsic_gaus=extrinsic, fov_rad_eg3d=fov,
                                                            fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)

            target_col_op = decoded_col_op.clone().detach()
            target_col_op[:, 0] = torch.min(target_col_op[:, 0],
                                            inverse_sigmoid(torch.ones_like(target_col_op[:, 0]) * 0.01))

            loss = (target_col_op - decoded_col_op).square().mean()
            progress_bar.set_description(f"Resetting opacity: MSE = {loss * 100:0.2f}")
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

        progress_bar = tqdm(range(1500), position=0, leave=True, disable=False)
        for _ in progress_bar:
            fov_deg = np.random.rand() * 5 + 13
            fov = fov_deg / 360 * 2 * np.pi
            extrinsic = self.get_random_extrinsic()
            z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

            decoded_rot_scale = self.eg3d_model.forward_rot_scale(z, w=None, extrinsic_eg3d=extrinsic,
                                                                  extrinsic_gaus=extrinsic, fov_rad_eg3d=fov,
                                                                  fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)
            target_scales = self.gaussians.target_scales * 0 + (-0.5 * math.log(30000))
            target_rot_scale = torch.concatenate((self.gaussians.target_rots, target_scales), dim=1).detach()

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


# class SceneAlignment:
#     gaussians: GaussianModel

#     def __init__(self, eg3d_model, gaussians: GaussianModel, overfit_single_id: bool = False, flame_init: bool = False):
#         self.gaussians = gaussians
#         self.eg3d_model = eg3d_model.to("cuda")
#         self.overfit_single_id = overfit_single_id
#         self.flame_init = flame_init

#         self.cameras_extent = 1
#         self.radius = 2.7
#         self.camera_lookat_point = torch.tensor([0, 0, 0], device="cuda")
#         self.z = torch.randn(1, 512).to(device="cuda").float()

#         # init deca decoder
#         E_flame = ResnetEncoder(outsize=236)
#         ckpt = torch.load("/home/beckmann/Projects/DECA/data/deca_model.tar")
#         copy_state_dict(E_flame.state_dict(), ckpt["E_flame"])
#         E_flame.to("cuda")
#         E_flame.eval()

#         self.E_flame = E_flame
#         self.flamelayer = FLAME(get_config()).to("cuda")
#         self.fan = FAN()

#         self.n_list = []
#         self._init_xyz()

#     def shift_and_set(self, x_shift, y_shift, z_shift, scale):
#         fov_deg = 17
#         fov = fov_deg / 360 * 2 * np.pi

#         current_xyz = self._xyz.detach().clone()

#         current_xyz *= scale
#         current_xyz[:, 0] += x_shift
#         current_xyz[:, 1] += y_shift
#         current_xyz[:, 2] += z_shift

#         self.eg3d_model._xyz = torch.nn.Parameter(current_xyz)

#         _, _ = self.eg3d_model(self.z, w=None, extrinsic_eg3d=self.extrinsic, extrinsic_gaus=self.extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

#         pos = self.eg3d_model._xyz.detach().cpu().numpy()
#         colors = self.eg3d_model.gaussian_model._features_dc.detach().cpu().numpy()[:, 0, :]
#         opacity = self.eg3d_model.gaussian_model._opacity.detach().cpu().numpy()*100

#         self.gaussians.create_from_pos_col(positions=pos, colors=colors, opacity=opacity)


#     def get_random_extrinsic(self, horizontal_stddev=np.pi * 0.3, vertical_stddev=np.pi * 0.25):
#         return UniformCameraPoseSampler.sample(
#             horizontal_stddev=horizontal_stddev,
#             vertical_stddev=vertical_stddev,
#             radius=self.radius,
#             device="cuda"
#         )

#     def get_camera_and_target(self, fov_deg=17, extrinsic=None, size=512, xyz=None):
#         fov = fov_deg / 360 * 2 * np.pi

#         if extrinsic is None:
#             extrinsic = self.get_random_extrinsic()
#         viewpoint = CustomCam(size=size, fov=fov, extr=extrinsic[0])
#         if not self.overfit_single_id:
#             self.z = torch.randn(1, 512).to(device="cuda").float()
#         target_image, decoded_features = self.eg3d_model(self.z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)
#         self.target_image = target_image
#         self.fov = fov
#         self.extrinsic = extrinsic

#         return viewpoint, target_image, decoded_features

#     def _init_xyz(self):
#         fov_deg = 17
#         fov = fov_deg / 360 * 2 * np.pi
#         extrinsic = self.get_random_extrinsic(0, 0)
#         self.extrinsic = extrinsic

#         final_positions = []
#         final_colors = []
#         num_heads = 1 if self.overfit_single_id else 1
#         with torch.no_grad():
#             for _ in range(num_heads):
#                 self.eg3d_model._init_xyz()
#                 z = self.z

#                 target_image, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)
#                 self.target_image = target_image

#                 if self.flame_init:
#                     self.init_xyz_from_flame(target_image)
#                     _, _ = self.eg3d_model(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov)

#                 pos = self.eg3d_model._xyz.detach().cpu().numpy()
#                 colors = self.eg3d_model.gaussian_model._features_dc.detach().cpu().numpy()[:, 0, :]
#                 opacity = self.eg3d_model.gaussian_model._opacity.detach().cpu().numpy()*100

#                 final_positions.append(pos)
#                 final_colors.append(colors)

#             final_positions = np.concatenate(final_positions)
#             final_colors = np.concatenate(final_colors)

#         self.gaussians.create_from_pos_col(positions=final_positions, colors=final_colors, opacity=opacity)


#     def init_xyz_from_flame(self, image):
#         E_flame = self.E_flame


#         # make exactly like flame
#         save_img = image.permute(1, 2, 0).detach().cpu().numpy()*255
#         import cv2

#         save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
#         cv2.imwrite("save_img.png", save_img.astype(int))

#         image = np.array(imread("save_img.png"))
#         h, w, _ = image.shape
#         resolution_inp = 224

#         det = self.fan
#         bbox, bbox_type = det.run(image.astype(np.uint8))
#         try:
#             left = bbox[0]; right=bbox[2]
#             top = bbox[1]; bottom=bbox[3]
#         except:
#             print("no face detected! run original image")
#             left = 0; right = h-1; top=0; bottom=w-1

#         old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
#         size = int(old_size*1.25)
#         src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
#         DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
#         tform = estimate_transform('similarity', src_pts, DST_PTS)

#         image = image/255.

#         dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
#         dst_image = dst_image.transpose(2,0,1)

#         image = torch.tensor(dst_image).float().cuda()

#         flame_out = E_flame(image.unsqueeze(0))
#         shape = flame_out[:, :100]
#         exp = flame_out[:, 150:200]
#         pose = flame_out[:, 200:206]
#         # cam = flame_out[:, 206:209]

#         pose[:, :3] = 0 # neutralize pose

#         flamelayer = self.flamelayer
#         faces = flamelayer.faces_tensor

#         vertices, _, _ = flamelayer(shape, exp, pose)
#         vertices = vertices.squeeze(0)

#         xyz = torch.zeros((faces.shape[0], 3)).cuda()
#         normals = torch.zeros((faces.shape[0], 3)).cuda()

#         xyz = []
#         normals = []
#         rots = []
#         scales = []
#         for i in range(faces.shape[0]):
#             triangle_area = self.compute_triangle_area(vertices[faces[i], :])
#             n = self.triangle_area_to_n(triangle_area)
#             self.n_list.append(n)
#             face_points = self.generate_points_on_triangle(vertices[faces[i], :], n)
#             xyz.append(face_points)
#             rot, normal, scale = self._get_triangle_info(vertices[faces[i], :])
#             normals += [normal.unsqueeze(0)]*n
#             rots += [rot.unsqueeze(0)]*n
#             scales += [scale.unsqueeze(0) / n]*n

#         xyz = torch.concatenate(xyz, dim = 0).cuda()
#         normals = torch.concatenate(normals, dim = 0).cuda()

#         stack = [xyz]
#         for factor in np.linspace(0, 0.075, 10)[1:]:
#             new = xyz + factor * normals
#             stack.append(new)

#         # xyz = torch.concatenate(stack, dim=0)

#         # xyz = xyz*2.5
#         # xyz[:, 1] += 1/37
#         # xyz[:, 2] += 0.075

#         self._xyz = xyz

#         self.eg3d_model._xyz = torch.nn.Parameter(xyz)
#         self.eg3d_model._xyz.requires_grad_(True)


#     def get_flame_verts(self, image):
#         with torch.no_grad():
#             E_flame = self.E_flame

#             # make exactly like flame
#             image = image.permute(1, 2, 0).detach().cpu().numpy()*255
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             h, w, _ = image.shape
#             resolution_inp = 224

#             det = self.fan
#             bbox, bbox_type = det.run(image.astype(np.uint8))
#             try:
#                 left = bbox[0]; right=bbox[2]
#                 top = bbox[1]; bottom=bbox[3]
#             except:
#                 print("no face detected! run original image")
#                 left = 0; right = h-1; top=0; bottom=w-1

#             old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
#             size = int(old_size*1.25)
#             src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
#             DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
#             tform = estimate_transform('similarity', src_pts, DST_PTS)

#             image = image/255.

#             dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
#             dst_image = dst_image.transpose(2,0,1)

#             image = torch.tensor(dst_image).float().cuda()

#             flame_out = E_flame(image.unsqueeze(0))
#             shape = flame_out[:, :100]
#             exp = flame_out[:, 150:200]
#             pose = flame_out[:, 200:206]

#             pose[:, :3] = 0 # neutralize pose

#             flamelayer = self.flamelayer
#             faces = flamelayer.faces_tensor

#             vertices, _, _ = flamelayer(shape, exp, pose)
#             vertices = vertices.squeeze(0)

#             xyz = torch.zeros((faces.shape[0], 3)).cuda()

#             xyz = []
#             for i in range(faces.shape[0]):
#                 n = self.n_list[i]
#                 face_points = self.generate_points_on_triangle(vertices[faces[i], :], n)
#                 xyz.append(face_points)

#             xyz = torch.concatenate(xyz, dim = 0).cuda()

#         return xyz

#         # self.eg3d_model._xyz = torch.nn.Parameter(xyz)
#         # shift_scale_dict = {
#         #     "scale": scale,
#         #     "x": x_shift,
#         #     "y": y_shift,
#         #     "z": z_shift
#         # }
#         # xyz *= scale
#         # xyz[:, 0] += x_shift
#         # xyz[:, 1] += y_shift
#         # xyz[:, 2] += z_shift

#         # fov = 17 / 360 * 2 * np.pi
#         # _, decoded_features = self.eg3d_model(self.z, w=None, extrinsic_eg3d=self.extrinsic, extrinsic_gaus=self.extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, shift_scale_dict=shift_scale_dict)

#         # return decoded_features, xyz


#     def _get_triangle_info(self, vertices):
#         v1 = vertices[0]
#         v2 = vertices[1]
#         v3 = vertices[2]

#         # Compute normal vector n
#         n = F.normalize(torch.cross(v2 - v1, v3 - v1), dim=0)

#         # Compute r2 as the normalized vector from the centroid to v1
#         m = (v1 + v2 + v3) / 3.0
#         r2 = F.normalize(v1 - m, dim=0)

#         # Compute r3 using the Gram-Schmidt process
#         r1 = n
#         r3 = F.normalize(torch.cross(r1, r2), dim=0)

#         # Rotation matrix R
#         R = torch.stack([r1, r2, r3], dim=1)
#         r = self.rotation_matrix_to_quaternion(R)

#         # Scaling vector S
#         s2 = torch.norm(m - v1)
#         s3 = torch.dot((v2 - m), r3)
#         S = torch.tensor([1e-3, s2, s3])  # Assuming s1 is a small constant for numerical stability

#         return r, n, S

#     def rotation_matrix_to_quaternion(self, R):
#         # Make sure the input matrix is of float type for precision
#         R = R.float()

#         # Preallocate the quaternion tensor
#         q = torch.zeros(4)

#         # Compute the trace of the matrix
#         trace = torch.trace(R)

#         if trace > 0:
#             s = torch.sqrt(trace + 1.0) * 2
#             q[0] = 0.25 * s
#             q[1] = (R[2, 1] - R[1, 2]) / s
#             q[2] = (R[0, 2] - R[2, 0]) / s
#             q[3] = (R[1, 0] - R[0, 1]) / s
#         elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
#             s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
#             q[0] = (R[2, 1] - R[1, 2]) / s
#             q[1] = 0.25 * s
#             q[2] = (R[0, 1] + R[1, 0]) / s
#             q[3] = (R[0, 2] + R[2, 0]) / s
#         elif R[1, 1] > R[2, 2]:
#             s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
#             q[0] = (R[0, 2] - R[2, 0]) / s
#             q[1] = (R[0, 1] + R[1, 0]) / s
#             q[2] = 0.25 * s
#             q[3] = (R[1, 2] + R[2, 1]) / s
#         else:
#             s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
#             q[0] = (R[1, 0] - R[0, 1]) / s
#             q[1] = (R[0, 2] + R[2, 0]) / s
#             q[2] = (R[1, 2] + R[2, 1]) / s
#             q[3] = 0.25 * s

#         # Normalize the quaternion to ensure it's a unit quaternion
#         q = q / torch.norm(q)
#         return q

#     def _get_face_normals(self, vertices):
#         # mean position
#         T = vertices.mean(1)

#         # move to origin
#         A = vertices[:, 0] - T
#         B = vertices[:, 1] - T
#         C = vertices[:, 2] - T

#         # direction vector of A and B
#         direction_vec = B - A
#         direction_vec = direction_vec / direction_vec.norm(p=2, dim=1, keepdim=True)

#         # normal vector
#         normal_vec = torch.cross(input=B-A, other=C-A)
#         normal_vec = normal_vec / normal_vec.norm(p=2, dim=1, keepdim=True)

#         return normal_vec

#     def generate_points_on_triangle(self, vertices, n):
#         """
#         Generate n points on the face of a triangle using PyTorch.

#         Parameters:
#         - vertices: A (3, 3) tensor containing the xyz coordinates of the triangle's vertices.
#         - n: The number of points to generate.

#         Returns:
#         - A (n, 3) tensor containing the xyz coordinates of the generated points.
#         """
#         points = torch.zeros((n, 3))
#         for i in range(n):
#             # Generate random barycentric coordinates
#             s = torch.rand(2)
#             s, _ = torch.sort(s)  # Ensure the generated values are in ascending order
#             t1, t2 = s[0], s[1] - s[0]
#             t3 = 1 - s[1]

#             # Convert barycentric coordinates to Cartesian coordinates
#             point = t1 * vertices[0] + t2 * vertices[1] + t3 * vertices[2]
#             points[i] = point

#         return points

#     def compute_triangle_area(self, vertices):
#         # Compute vectors of two sides of the triangle
#         side1 = vertices[1] - vertices[0]
#         side2 = vertices[2] - vertices[0]

#         # Compute cross product of the two sides
#         cross_product = torch.cross(side1, side2)

#         # Compute area of the triangle using half of the magnitude of the cross product
#         area = 0.5 * torch.norm(cross_product)

#         return area

#     def triangle_area_to_n(self, face_space, min=6.225e-9, max=0.0002):
#         n = (face_space.item() - min) / (max - min)
#         n *= 9
#         n += 1

#         return round(n)

#     # init utils
#     def compute_radial_gradients(self, cube):
#         cube_size = cube.shape[0]
#         center = np.array([cube_size // 2, cube_size // 2, cube_size // 2])

#         # Compute the gradient using np.gradient
#         gradients = np.array(np.gradient(cube))

#         # Initialize an empty array for radial gradient magnitudes
#         radial_grad_magnitude = np.zeros(cube.shape)

#         for x in range(cube_size):
#             for y in range(cube_size):
#                 for z in range(cube_size):
#                     # Compute the direction vector from the center to the current point
#                     direction_vector = np.array([x, y, z]) - center
#                     direction_vector_norm = np.linalg.norm(direction_vector)
#                     if direction_vector_norm > 0:  # Avoid division by zero
#                         direction_vector_normalized = direction_vector / direction_vector_norm

#                         # Compute the gradient vector at the current point
#                         gradient_vector = gradients[:, x, y, z]

#                         # Project the gradient vector onto the direction vector
#                         radial_gradient = np.dot(gradient_vector, direction_vector_normalized)

#                         # Store the magnitude of the radial gradient (projection result)
#                         radial_grad_magnitude[x, y, z] = abs(radial_gradient)  # Use abs to consider magnitude only

#         return radial_grad_magnitude

#     # resetting
#     def reset_col_op_decoder(self, gaussian_optimizer, xyz=None):
#         optim = torch.optim.Adam(self.eg3d_model.color_opacity_decoder.parameters(), lr=0.005)

#         progress_bar = tqdm(range(5000), position=0, leave=True, disable=False)
#         for _ in progress_bar:
#             fov_deg = np.random.rand() * 5 + 13
#             fov = fov_deg / 360 * 2 * np.pi
#             extrinsic = self.get_random_extrinsic()
#             z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

#             decoded_col_op = self.eg3d_model.forward_col_op(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)

#             target_col_op = decoded_col_op.clone().detach()
#             target_col_op[:, 0] = torch.min(target_col_op[:, 0], inverse_sigmoid(torch.ones_like(target_col_op[:, 0])*0.01))

#             loss = (target_col_op - decoded_col_op).square().mean()
#             progress_bar.set_description(f"Resetting opacity: MSE = {loss*100:0.2f}")
#             loss.backward()

#             optim.step()
#             optim.zero_grad()

#         # reset optimizer
#         for param_group in gaussian_optimizer.param_groups:
#             # Check if this is the opacity_decoder group
#             if 'name' in param_group and param_group['name'] == 'color_opacity_decoder':
#                 for param in param_group['params']:
#                     # Check if this param is in the optimizer state (it should be if it has gradients)
#                     if param in gaussian_optimizer.state:
#                         # Reset first and second moments
#                         gaussian_optimizer.state[param]['exp_avg'].zero_()
#                         gaussian_optimizer.state[param]['exp_avg_sq'].zero_()
#         print("Done resetting {}. Continue normal training".format("opacity"))

#         torch.cuda.empty_cache()

#         return gaussian_optimizer

#     def init_rot_scale_decoder(self, gaussian_optimizer, xyz=None):
#         optim = torch.optim.Adam(self.eg3d_model.scaling_rotation_decoder.parameters(), lr=0.002)

#         progress_bar = tqdm(range(500), position=0, leave=True, disable=False)
#         for _ in progress_bar:
#             fov_deg = np.random.rand() * 5 + 13
#             fov = fov_deg / 360 * 2 * np.pi
#             extrinsic = self.get_random_extrinsic()
#             z = torch.randn(1, 512).to(device="cuda").float() if not self.overfit_single_id else self.z

#             decoded_rot_scale = self.eg3d_model.forward_rot_scale(z, w=None, extrinsic_eg3d=extrinsic, extrinsic_gaus=extrinsic, fov_rad_eg3d=fov, fov_rad_gaus=fov, xyz=xyz, only_render_eg3d=True)
#             target_rot_scale = torch.concatenate((self.gaussians.target_rots, self.gaussians.target_scales), dim=1).detach()

#             loss = (target_rot_scale - decoded_rot_scale).square().mean()
#             progress_bar.set_description(f"Init rotscale: MSE = {loss:0.2f}")
#             loss.backward()

#             optim.step()
#             optim.zero_grad()

#         # reset optimizer
#         for param_group in gaussian_optimizer.param_groups:
#             # Check if this is the opacity_decoder group
#             if 'name' in param_group and param_group['name'] == 'scaling_rotation_deocder':
#                 for param in param_group['params']:
#                     # Check if this param is in the optimizer state (it should be if it has gradients)
#                     if param in gaussian_optimizer.state:
#                         # Reset first and second moments
#                         gaussian_optimizer.state[param]['exp_avg'].zero_()
#                         gaussian_optimizer.state[param]['exp_avg_sq'].zero_()
#         print("Done resetting {}. Continue normal training".format("opacity"))

#         torch.cuda.empty_cache()

#         return gaussian_optimizer


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] - camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn