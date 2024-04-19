import torch
import numpy as np
import cv2


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
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
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

        # xyz = torch.concatenate(stack, dim=0)

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

        # xyz = torch.concatenate((xyz, shoulder, walls), dim = 0)

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