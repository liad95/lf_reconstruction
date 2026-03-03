import os

import cupy as cp
import numpy as np
import scipy
import torch
import torch.nn.functional as interpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.sparse import linalg

from globals import LF_FOLDER, MASK_FOLDER


# region interpolations
def find_phase_mask_locations_gpu(X, Y, SinX, SinY, L):
    """
    Returns the locations in the phase mask given the LF locations and angles, on GPU
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :return: the locations in the phase mask
    """
    SinZ = torch.sqrt(1 - torch.pow(SinX, 2) - torch.pow(SinY, 2))
    inter1_points_x = X + L * SinX / SinZ
    inter1_points_y = Y + L * SinY / SinZ
    return inter1_points_x, inter1_points_y


def find_phase_mask_locations(X, Y, SinX, SinY, L):
    """
    Returns the locations in the phase mask given the LF locations and angles
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :return: the locations in the phase mask
    """
    SinZ = np.sqrt(1 - np.power(SinX, 2) - np.power(SinY, 2))
    inter1_points_x = X + L * SinX / SinZ
    inter1_points_y = Y + L * SinY / SinZ
    return inter1_points_x, inter1_points_y


def find_mask_angles_gpu2(inter1_points_x, inter1_points_y, maskx, masky, sampling_dist_mask_plane, method=''):
    """
    Interpolates the sines changes from the phase mask given the phase mask locations and angles, on GPU
    :param inter1_points_x: meshgrid of relevant phase mask X locations
    :param inter1_points_y: meshgrid of relevant phase mask Y locations
    :param maskx: the X element of the phase mask gradient
    :param masky: the Y element of the phase mask gradient
    :param sampling_dist_mask_plane: the sampling distance of the phase mask plane
    :param method: the method of interpolation from the phase mask, the default is 'linear'
    :return: the sines changes from the phase mask
    """
    delta_sin_size = maskx.shape
    points_shape = inter1_points_x.shape
    inter1_points_x = inter1_points_x * 2 / ((delta_sin_size[0] - 1) * sampling_dist_mask_plane)
    inter1_points_y = inter1_points_y * 2 / ((delta_sin_size[0] - 1) * sampling_dist_mask_plane)
    inter1_points_x = inter1_points_x.reshape(points_shape[0] * points_shape[1], points_shape[2] * points_shape[3])
    inter1_points_y = inter1_points_y.reshape(points_shape[0] * points_shape[1], points_shape[2] * points_shape[3])
    interp1_points = torch.stack((inter1_points_x, inter1_points_y), dim=2).unsqueeze(0)
    maskx = maskx.unsqueeze(0).unsqueeze(0)
    masky = masky.unsqueeze(0).unsqueeze(0)

    delta_sin_x_interp1 = interpolator.grid_sample(maskx, interp1_points, mode=method, padding_mode='zeros',
                                                   align_corners=True)
    delta_sin_y_interp1 = interpolator.grid_sample(masky, interp1_points, mode=method, padding_mode='zeros',
                                                   align_corners=True)
    delta_sin_x_interp1 = delta_sin_x_interp1.squeeze().reshape(points_shape)
    delta_sin_y_interp1 = delta_sin_y_interp1.squeeze().reshape(points_shape)
    return delta_sin_x_interp1, delta_sin_y_interp1


def find_mask_angles_gpu(inter1_points_x, inter1_points_y, maskx, masky, sampling_dist_mask_plane, method=''):
    """
    Interpolates the sines changes from the phase mask given the phase mask locations and angles, on GPU
    :param inter1_points_x: meshgrid of relevant phase mask X locations
    :param inter1_points_y: meshgrid of relevant phase mask Y locations
    :param maskx: the X element of the phase mask gradient
    :param masky: the Y element of the phase mask gradient
    :param sampling_dist_mask_plane: the sampling distance of the phase mask plane
    :param method: the method of interpolation from the phase mask, the default is 'linear'
    :return: the sines changes from the phase mask
    """
    delta_sin_size = maskx.shape
    inter1_points_x = inter1_points_x * 2 / ((delta_sin_size[0] - 1) * sampling_dist_mask_plane)
    inter1_points_y = inter1_points_y * 2 / ((delta_sin_size[0] - 1) * sampling_dist_mask_plane)
    interp1_points = torch.stack((inter1_points_x, inter1_points_y), dim=2).unsqueeze(0)
    maskx = maskx.unsqueeze(0).unsqueeze(0)
    masky = masky.unsqueeze(0).unsqueeze(0)

    delta_sin_x_interp1 = interpolator.grid_sample(maskx, interp1_points, mode=method, padding_mode='zeros',
                                                   align_corners=True)
    delta_sin_y_interp1 = interpolator.grid_sample(masky, interp1_points, mode=method, padding_mode='zeros',
                                                   align_corners=True)
    return delta_sin_x_interp1.squeeze(), delta_sin_y_interp1.squeeze()


def find_mask_angles(inter1_points_x, inter1_points_y, maskx, masky, sampling_dist_mask_plane, method='linear'):
    """
    Interpolates the sines changes from the phase mask given the phase mask locations and angles
    :param inter1_points_x: meshgrid of relevant phase mask X locations
    :param inter1_points_y: meshgrid of relevant phase mask Y locations
    :param maskx: the X element of the phase mask gradient
    :param masky: the Y element of the phase mask gradient
    :param sampling_dist_mask_plane: the sampling distance of the phase mask plane
    :param method: the method of interpolation from the phase mask, the default is 'linear'
    :return: the sines changes from the phase mask
    """
    delta_sin_size = maskx.shape
    x_delta_sin = np.linspace(0, delta_sin_size[1] * sampling_dist_mask_plane, delta_sin_size[1],
                              endpoint=False) - \
                  (delta_sin_size[1] - 1) * sampling_dist_mask_plane / 2
    y_delta_sin = np.linspace(0, delta_sin_size[0] * sampling_dist_mask_plane, delta_sin_size[0],
                              endpoint=False) - \
                  (delta_sin_size[0] - 1) * sampling_dist_mask_plane / 2

    inter1_points = np.array([inter1_points_x.ravel(), inter1_points_y.ravel()]).T
    delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), maskx, bounds_error=False, method=method)
    delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), masky, bounds_error=False, method=method)
    delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(inter1_points_x.shape)
    delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(inter1_points_x.shape)
    return delta_sin_x_interp1, delta_sin_y_interp1


def find_forward_locations(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1):
    """
    Interpolates the locations for the forward warping given sines changes from the phase mask, and the LF locations and angles
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :param delta_sin_x_interp1: meshgrid of the X element of the sines changes
    :param delta_sin_y_interp1: meshgrid of the Y element of the sines changes
    :return: the locations for the forward warping
    """
    SinZ = np.sqrt(1 - np.power(SinX, 2) - np.power(SinY, 2))
    inter2_points_sinx = SinX - delta_sin_x_interp1
    inter2_points_siny = SinY - delta_sin_y_interp1
    sinz_squared = 1 - np.power(inter2_points_sinx, 2) - np.power(inter2_points_siny, 2)
    sinz_squared = np.where(sinz_squared <= 0, 0.01, sinz_squared)
    sinz = np.sqrt(sinz_squared)
    inter2_points_x = X - L * (inter2_points_sinx / sinz - SinX / SinZ)
    inter2_points_y = Y - L * (inter2_points_siny / sinz - SinY / SinZ)
    return inter2_points_x, inter2_points_y


def find_forward_locations_gpu(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1):
    """
    Interpolates the locations for the forward warping given sines changes from the phase mask, and the LF locations and angles, on GPU
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :param delta_sin_x_interp1: meshgrid of the X element of the sines changes
    :param delta_sin_y_interp1: meshgrid of the Y element of the sines changes
    :return: the locations for the forward warping
    """
    SinZ = torch.sqrt(1 - torch.pow(SinX, 2) - torch.pow(SinY, 2))
    inter2_points_sinx = SinX - delta_sin_x_interp1
    inter2_points_siny = SinY - delta_sin_y_interp1
    sinz = torch.sqrt(1 - torch.pow(inter2_points_sinx, 2) - torch.pow(inter2_points_siny, 2))
    inter2_points_x = X - L * (inter2_points_sinx / sinz - SinX / SinZ)
    inter2_points_y = Y - L * (inter2_points_siny / sinz - SinY / SinZ)
    return inter2_points_x, inter2_points_y


def find_forward_locations_gpu_parallel2_and(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1, deltas_x,
                                             deltas_y):
    """
    Interpolates the locations for the forward warping given sines changes from the phase mask, and the LF locations and angles, on GPU
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :param delta_sin_x_interp1: meshgrid of the X element of the sines changes
    :param delta_sin_y_interp1: meshgrid of the Y element of the sines changes
    :return: the locations for the forward warping
    """
    deltas = torch.cartesian_prod(deltas_x, deltas_y)
    deltas = deltas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    SinX = SinX.unsqueeze(0)
    SinY = SinY.unsqueeze(0)
    delta_sin_x_interp1 = delta_sin_x_interp1.unsqueeze(0)
    delta_sin_y_interp1 = delta_sin_y_interp1.unsqueeze(0)
    SinZ = torch.sqrt(1 - torch.pow(SinX, 2) - torch.pow(SinY, 2))

    inter2_points_sinx_with_delta = SinX - delta_sin_x_interp1 + deltas[:, 0]
    inter2_points_siny_with_delta = SinY - delta_sin_y_interp1 + deltas[:, 1]

    sinz_delta = torch.sqrt(
        1 - torch.pow(inter2_points_sinx_with_delta, 2) - torch.pow(inter2_points_siny_with_delta, 2))

    inter2_points_x_delta = (X - L * (inter2_points_sinx_with_delta / sinz_delta - SinX / SinZ)) / torch.max(X)
    inter2_points_y_delta = (Y - L * (inter2_points_siny_with_delta / sinz_delta - SinY / SinZ)) / torch.max(Y)
    return (inter2_points_x_delta, inter2_points_y_delta)

def find_forward_locations_gpu_parallel2(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1, deltas):
    """
    Interpolates the locations for the forward warping given sines changes from the phase mask, and the LF locations and angles, on GPU
    :param X: meshgrid of X locations in the LF plane
    :param Y: meshgrid of Y locations in the LF plane
    :param SinX: meshgrid of X angles in the LF
    :param SinY: meshgrid of Y angles in the LF
    :param L: Distance between the LF plane and the phase mask
    :param delta_sin_x_interp1: meshgrid of the X element of the sines changes
    :param delta_sin_y_interp1: meshgrid of the Y element of the sines changes
    :return: the locations for the forward warping
    """
    deltas = deltas.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    SinX = SinX.unsqueeze(0)
    SinY = SinY.unsqueeze(0)
    delta_sin_x_interp1 = delta_sin_x_interp1.unsqueeze(0)
    delta_sin_y_interp1 = delta_sin_y_interp1.unsqueeze(0)
    SinZ = torch.sqrt(1 - torch.pow(SinX, 2) - torch.pow(SinY, 2))

    inter2_points_sinx_with_delta = SinX - (delta_sin_x_interp1 + deltas)
    inter2_points_siny_with_delta = SinY - (delta_sin_y_interp1 + deltas)
    inter2_points_sinx = SinX - delta_sin_x_interp1
    inter2_points_siny = SinY - delta_sin_y_interp1

    sinz_delta_x = torch.sqrt(1 - torch.pow(inter2_points_sinx_with_delta, 2) - torch.pow(inter2_points_siny, 2))
    sinz_delta_y = torch.sqrt(1 - torch.pow(inter2_points_siny_with_delta, 2) - torch.pow(inter2_points_sinx, 2))

    inter2_points_x_delta_x = (X - L * (inter2_points_sinx_with_delta / sinz_delta_x - SinX / SinZ)) / torch.max(X)
    inter2_points_y_delta_x = (Y - L * (inter2_points_siny / sinz_delta_x - SinY / SinZ)) / torch.max(Y)
    inter2_points_x_delta_y = (X - L * (inter2_points_sinx / sinz_delta_y - SinX / SinZ)) / torch.max(X)
    inter2_points_y_delta_y = (Y - L * (inter2_points_siny_with_delta / sinz_delta_y - SinY / SinZ)) / torch.max(Y)
    return (inter2_points_x_delta_x, inter2_points_y_delta_x), (inter2_points_x_delta_y, inter2_points_y_delta_y)
def point_in_grid(x, y, height, width):
    """
    checks whether the points are in the grid
    :param x: meshgrid of x locations
    :param y: meshgrid of y locations
    :param height: boundary height
    :param width: boundary width
    :return: matrix of boolean for x,y whether the points are in the grid
    """
    is_in_grid = np.logical_and.reduce([x >= 0, x < width, y >= 0, y < height])
    return is_in_grid


def point_in_grid_gpu(x, y, height, width):
    """
    checks whether the points are in the grid, on GPU
    :param x: meshgrid of x locations
    :param y: meshgrid of y locations
    :param height: boundary height
    :param width: boundary width
    :return: matrix of boolean for x,y whether the points are in the grid
    """
    is_in_grid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return is_in_grid


def create_weighted_forward(interp_x, interp_y, sampling_dist, matrix):
    """
    Warp a matrix to a standard grid with specified interpolation points, using forward warping
    :param interp_x: the x locations of interpolation of the matrix
    :param interp_y: the y locations of interpolation of the matrix
    :param sampling_dist: the sampling distance of the matrix
    :param matrix: the matrix to perform forward warping upon
    :return: forward warped matrix according to the interpolation points
    """
    weight_matrix = create_transform_matrix(interp_x, interp_y, sampling_dist, matrix.shape, matrix.shape)
    shape = matrix.shape
    matrix = matrix.flatten()
    weighted_matrix = (weight_matrix @ matrix).reshape(shape)
    return weighted_matrix


def create_transform_matrix(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    """
    Create a transformation matrix for forward warping. Interpolation is performed, and conversion to a new grid shape is possible using the phase_mask_shape param.
    :param interp_x: the x locations of interpolation of the matrix
    :param interp_y: the y locations of interpolation of the matrix
    :param sampling_dist: the sampling distance of the matrix
    :param phase_mask_shape: the shape of the new grid
    :param matrix_shape: the shape of the matrix grid
    :return: A transformation matrix for forward warping
    """
    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]

    # converting the locations to indexes
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist

    # finding the 4 neighboring indexes
    high_x = np.ceil(inter2_points_x)
    low_x = np.floor(inter2_points_x)
    high_y = np.ceil(inter2_points_y)
    low_y = np.floor(inter2_points_y)

    # checking whether the neighboring indexes are inside the LF grid
    is_valid = np.concatenate((point_in_grid(low_x, low_y, height_phase, width_phase),
                               point_in_grid(high_x, low_y, height_phase, width_phase),
                               point_in_grid(low_x, high_y, height_phase, width_phase),
                               point_in_grid(high_x, high_y, height_phase, width_phase)))

    # calculating the neighbors' weights
    upper_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    weights = np.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    # converting to flattened indexes
    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = np.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = np.squeeze(np.argwhere(is_valid))
    col_idx = np.tile(np.arange(height_matrix * width_matrix), 4)

    # creating the conversion matrix. size of HW_phase*HW_matrix, each column has max 4 weights (4 neighbors)
    weight_matrix = scipy.sparse.csr_array((weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                                           shape=(height_phase * width_phase, height_matrix * width_matrix))

    return weight_matrix


def create_transform_matrix_gpu(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    """
    Create a transformation matrix for forward warping. Interpolation is performed, and conversion to a new grid shape is possible using the phase_mask_shape param, on GPU
    :param interp_x: the x locations of interpolation of the matrix
    :param interp_y: the y locations of interpolation of the matrix
    :param sampling_dist: the sampling distance of the matrix
    :param phase_mask_shape: the shape of the new grid
    :param matrix_shape: the shape of the matrix grid
    :return: A transformation matrix for forward warping
    """

    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]

    # converting the locations to indexes
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist

    # finding the 4 neighboring indexes
    high_x = torch.ceil(inter2_points_x)
    low_x = torch.floor(inter2_points_x)
    high_y = torch.ceil(inter2_points_y)
    low_y = torch.floor(inter2_points_y)

    # checking whether the neighboring indexes are inside the LF grid
    is_valid = torch.concatenate((point_in_grid_gpu(low_x, low_y, height_phase, width_phase),
                                  point_in_grid_gpu(high_x, low_y, height_phase, width_phase),
                                  point_in_grid_gpu(low_x, high_y, height_phase, width_phase),
                                  point_in_grid_gpu(high_x, high_y, height_phase, width_phase)))

    # calculating the neighbors' weights
    upper_right_weight = (1 - torch.abs(high_x - inter2_points_x)) * (1 - torch.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - torch.abs(low_x - inter2_points_x)) * (1 - torch.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - torch.abs(high_x - inter2_points_x)) * (1 - torch.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - torch.abs(low_x - inter2_points_x)) * (1 - torch.abs(low_y - inter2_points_y))
    weights = torch.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    # converting to flattened indexes
    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = torch.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).int()
    valid_idx = torch.squeeze(torch.argwhere(is_valid))
    col_idx = torch.arange(height_matrix * width_matrix, device='cuda').tile((4,))
    idx = torch.stack((row_idx[valid_idx], col_idx[valid_idx]))

    # creating the conversion matrix. size of HW_phase*HW_matrix, each column has max 4 weights (4 neighbors)
    weight_matrix = torch.sparse_coo_tensor(idx, weights[valid_idx],
                                            size=(height_phase * width_phase, height_matrix * width_matrix))

    return weight_matrix


def create_transform_matrix_gpu2(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    """
    Create a transformation matrix for forward warping. Interpolation is performed, and conversion to a new grid shape is possible using the phase_mask_shape param, on GPU
    :param interp_x: the x locations of interpolation of the matrix
    :param interp_y: the y locations of interpolation of the matrix
    :param sampling_dist: the sampling distance of the matrix
    :param phase_mask_shape: the shape of the new grid
    :param matrix_shape: the shape of the matrix grid
    :return: A transformation matrix for forward warping
    """

    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]

    # converting the locations to indexes
    inter2_points_x = interp_x.flatten(end_dim=1)
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten(end_dim=1)
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist

    # finding the 4 neighboring indexes
    high_x = torch.ceil(inter2_points_x)
    low_x = torch.floor(inter2_points_x)
    high_y = torch.ceil(inter2_points_y)
    low_y = torch.floor(inter2_points_y)

    # checking whether the neighboring indexes are inside the LF grid
    is_valid = torch.concatenate((point_in_grid_gpu(low_x, low_y, height_phase, width_phase),
                                  point_in_grid_gpu(high_x, low_y, height_phase, width_phase),
                                  point_in_grid_gpu(low_x, high_y, height_phase, width_phase),
                                  point_in_grid_gpu(high_x, high_y, height_phase, width_phase)))

    # calculating the neighbors' weights
    upper_right_weight = (1 - torch.abs(high_x - inter2_points_x)) * (1 - torch.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - torch.abs(low_x - inter2_points_x)) * (1 - torch.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - torch.abs(high_x - inter2_points_x)) * (1 - torch.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - torch.abs(low_x - inter2_points_x)) * (1 - torch.abs(low_y - inter2_points_y))
    weights = torch.concatenate(
        (lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight)).flatten()

    i, j = torch.arange(matrix_shape[2]).reshape(-1, 1), torch.arange(matrix_shape[3]).reshape(1, -1)

    # Compute matrix

    # converting to flattened indexes
    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = torch.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx))
    row_idx = row_idx.int().flatten()
    valid_idx = torch.squeeze(torch.argwhere(is_valid.flatten()))
    col_idx = torch.arange(height_matrix * width_matrix * matrix_shape[2] * matrix_shape[3], device='cuda').tile((4,))
    idx = torch.stack((row_idx[valid_idx], col_idx[valid_idx]))

    # creating the conversion matrix. size of HW_phase*HW_matrix, each column has max 4 weights (4 neighbors)
    weight_matrix = torch.sparse_coo_tensor(idx, weights[valid_idx],
                                            size=(height_phase * width_phase,
                                                  height_matrix * width_matrix * matrix_shape[2] * matrix_shape[3]))

    return weight_matrix


def create_transform_matrix_gpu_weight_row_col(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    """
    Returns the column and row indexes and the corresponding weights of transformation matrix for forward warping. Interpolation is performed, and conversion to a new grid
    shape is possible using the phase_mask_shape param, on GPU :param interp_x: the x locations of interpolation of
    the matrix :param interp_y: the y locations of interpolation of the matrix :param sampling_dist: the sampling
    distance of the matrix :param phase_mask_shape: the shape of the new grid :param matrix_shape: the shape of the
    matrix grid :return: The column and row indexes and the corresponding weights (transformation matrix elements).
    This is for the creation of a larger sparse block matrix outside the function.
    """

    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]

    # converting the locations to indexes
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist

    # finding the 4 neighboring indexes
    high_x = cp.ceil(inter2_points_x)
    low_x = cp.floor(inter2_points_x)
    high_y = cp.ceil(inter2_points_y)
    low_y = cp.floor(inter2_points_y)

    # checking whether the neighboring indexes are inside the LF grid
    is_valid = cp.concatenate((point_in_grid_gpu(low_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(low_x, high_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, high_y, height_phase, width_phase)))

    # calculating the neighbors' weights
    upper_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    weights = cp.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    # converting to flattened indexes
    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = cp.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = cp.squeeze(cp.argwhere(is_valid))
    col_idx = cp.tile(cp.arange(height_matrix * width_matrix), 4)

    return weights[valid_idx], row_idx[valid_idx], col_idx[valid_idx]


# endregion

# region load_funcs

def load_lf_mask(name=None):
    """
    Loads a LF and a phase mask from .mat files
    :param name: the postfix of the file names
    :return: the LF and the phase mask
    """
    lf_file_name = 'lf_' + name + '.mat'
    mask_file_name = 'mask_' + name + '.mat'
    lf = load_mat_file(LF_FOLDER, lf_file_name)['I_LF']
    mask = load_mat_file(MASK_FOLDER, mask_file_name)['maskL']
    N_mask = int(np.sqrt(len(mask)))
    mask = np.reshape(mask, (N_mask, N_mask))
    return lf, mask


def load_lf(name):
    """
    Loads a LF from a .mat file
    :param name: the postfix of the file name
    :return: the LF
    """
    lf_file_name = 'lf_' + name + '.mat'
    lf = load_mat_file(LF_FOLDER, lf_file_name)['I_LF']
    return lf


def load_mat_file(folder_path, file_name):
    """
    Loads a .mat file in a specified folder
    :param folder_path: the path of the folder
    :param file_name: the file name of the .mat file to load
    :return: the loaded matrix
    """
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        mat_data = scipy.io.loadmat(file_path)
        print(f"Loaded {file_name} successfully.")
        return mat_data
    else:
        print(f"File {file_name} does not exist in the folder.")
        return None

# endregion

def calc_correlation(matrix1, matrix2):
    return np.sum(matrix1*matrix2)/(np.sqrt(np.sum(matrix1*matrix1))*np.sqrt(np.sum(matrix2*matrix2)))