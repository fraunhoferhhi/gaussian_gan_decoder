import numpy as np
import torch


def create_samples(samples_per_axis=256, voxel_origin=[0, 0, 0], cube_length=1.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (samples_per_axis - 1)

    overall_index = torch.arange(0, samples_per_axis ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(samples_per_axis ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % samples_per_axis
    samples[:, 1] = (overall_index.float() / samples_per_axis) % samples_per_axis
    samples[:, 0] = ((overall_index.float() / samples_per_axis) / samples_per_axis) % samples_per_axis

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = samples_per_axis ** 3
    return samples.unsqueeze(0), num_samples


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
        x0, y0, z0 = int(torch.floor(x)), int(torch.floor(y)), int(torch.floor(z))
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