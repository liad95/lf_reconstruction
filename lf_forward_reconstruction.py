import os

import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import scipy
from scipy.sparse import linalg
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from lf_reconstructor import lf_reconstructor
import math
from utils import *


class lf_forward_reconstructor(lf_reconstructor):

    def __init__(self, max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N, L, angle_finder,
                 lf_shape):
        super().__init__(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N, L, angle_finder,
                         lf_shape)
        self._create_lf_grid()
    def reconstruct_lf_with_gradient(self, lf, gradientx, gradienty):
        self.lf = lf
        self.delta_sin_x = gradientx
        self.delta_sin_y = gradienty
        self._find_mask_location_points()
        self._get_interpolation_points()
        reconstructed_lf = self._forward_warp_lf_adv2()
        return reconstructed_lf
    def reconstruct_lf(self, lf, mask):
        self.lf = lf
        self.mask = mask
        self.delta_sin_x, self.delta_sin_y = self.angle_finder.get_delta_sin(self.mask)

        self._find_mask_location_points()
        self._get_interpolation_points()
        reconstructed_lf = self._forward_warp_lf_adv2()
        return reconstructed_lf

    def _find_mask_location_points(self):
        self.inter1_points_x = self.X + self.L * self.SinX / self.SinZ
        self.inter1_points_y = self.Y + self.L * self.SinY / self.SinZ

    def _get_interpolation_points(self):
        delta_sin_size = self.delta_sin_x.shape
        x_delta_sin = np.linspace(0, delta_sin_size[1] * self.sampling_dist_mask_plane, delta_sin_size[1],
                                  endpoint=False) - \
                      (delta_sin_size[1] - 1) * self.sampling_dist_mask_plane / 2
        y_delta_sin = np.linspace(0, delta_sin_size[0] * self.sampling_dist_mask_plane, delta_sin_size[0],
                                  endpoint=False) - \
                      (delta_sin_size[0] - 1) * self.sampling_dist_mask_plane / 2

        inter1_points = np.array([self.inter1_points_x.ravel(), self.inter1_points_y.ravel()]).T
        delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), self.delta_sin_x, bounds_error=False)
        delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), self.delta_sin_y, bounds_error=False)
        delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(self.inter1_points_x.shape)
        delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(self.inter1_points_x.shape)

        # endregion
        # region part 2
        inter2_points_sinx = self.SinX - delta_sin_x_interp1
        inter2_points_siny = self.SinY - delta_sin_y_interp1
        sinz = np.sqrt(1 - np.power(inter2_points_sinx, 2) - np.power(inter2_points_siny, 2))
        self.inter2_points_x = self.X - self.L * (inter2_points_sinx / sinz - self.SinX / self.SinZ)
        self.inter2_points_y = self.Y - self.L * (inter2_points_siny / sinz - self.SinY / self.SinZ)

    def _forward_warp_lf(self):
        lf_reconstructed = np.zeros((self.height, self.width, self.n_angles, self.n_angles))


        for i in range(self.n_angles):
            for j in range(self.n_angles):
                lf = self.lf[:, :, i, j].flatten()
                inter2_points_x = self.inter2_points_x[:, :, i, j].flatten()
                inter2_points_y = self.inter2_points_y[:, :, i, j].flatten()

                for k in range(len(lf)):
                    indexes, weights = self.find_weights(inter2_points_x[k], inter2_points_y[k])
                    lf_reconstructed[indexes[1],indexes[0], i,j] += lf[k] * weights
                    if (k % 1000 == 0):
                        print(k+ k*j+k*j*i)
        return lf_reconstructed

    def _create_lf_grid(self):
        self.sin_x = np.linspace(-self.max_sin+1/14, self.max_sin-1/14, self.n_angles)
        self.sin_y = np.linspace(-self.max_sin+1/14, self.max_sin-1/14, self.n_angles)
        self.x = np.linspace(0, self.width * self.sampling_dist_lf_plane,
                             self.width, endpoint=False) - (self.width - 1) * self.sampling_dist_lf_plane / 2
        self.y = np.linspace(0, self.height * self.sampling_dist_lf_plane,
                             self.height, endpoint=False) - (self.height - 1) * self.sampling_dist_lf_plane / 2
        self.X, self.Y, self.SinX, self.SinY = np.meshgrid(self.x, self.y, self.sin_x, self.sin_x, indexing='ij')
        self.SinZ = np.sqrt(1 - np.power(self.SinX, 2) - np.power(self.SinY, 2))

    def find_weights(self, x, y):
        x_idx = (x + self.width * self.sampling_dist_lf_plane / 2) / self.sampling_dist_lf_plane
        y_idx = (y + self.width * self.sampling_dist_lf_plane / 2) / self.sampling_dist_lf_plane
        x_ceil = math.ceil(x_idx)
        x_floor = math.floor(x_idx)
        y_ceil = math.ceil(y_idx)
        y_floor = math.floor(y_idx)
        points_x, points_y = np.meshgrid(np.array([x_floor, x_ceil]), np.array([y_floor, y_ceil]))
        point_valid = self._point_in_grid(points_x, points_y)

        weights = (1 - np.abs(points_x - x_idx)) * (1 - np.abs(points_y - y_idx)) * point_valid
        indexes_x = points_x * point_valid
        indexes_y = points_y * point_valid
        return (indexes_x, indexes_y), weights

    def _point_in_grid(self, x, y):
        # is_in_grid = np.zeros_like(x, dtype=bool)
        is_in_grid = np.logical_and.reduce([x >= 0, x < self.width, y >= 0, y < self.height])
        return is_in_grid

    def _forward_warp_lf_adv2(self):
        lf_reconstructed = np.zeros_like(self.lf)
        for i in range(self.n_angles):
            for j in range(self.n_angles):
                lf_reconstructed[:, :, i, j] = create_weighted_forward(self.inter2_points_x[:, :, i, j],
                                                                       self.inter2_points_y[:, :, i, j],
                                                                       self.sampling_dist_lf_plane, self.lf[:, :, i, j])
        return lf_reconstructed

    def _forward_warp_lf_adv(self):
        lf_reconstructed = np.zeros_like(self.lf)
        for i in range(self.n_angles):
            for j in range(self.n_angles):
                inter2_points_x = self.inter2_points_x[:, :, i, j].flatten()
                inter2_points_x = (
                                          inter2_points_x + self.width * self.sampling_dist_lf_plane / 2) / self.sampling_dist_lf_plane
                inter2_points_y = self.inter2_points_y[:, :, i, j].flatten()
                inter2_points_y = (
                                          inter2_points_y + self.height * self.sampling_dist_lf_plane / 2) / self.sampling_dist_lf_plane
                high_x = np.ceil(inter2_points_x)
                low_x = np.floor(inter2_points_x)
                high_y = np.ceil(inter2_points_y)
                low_y = np.floor(inter2_points_y)

                is_valid = np.concatenate((self._point_in_grid(low_x, low_y), self._point_in_grid(high_x, low_y),
                                           self._point_in_grid(low_x, high_y), self._point_in_grid(high_x, high_y)))
                upper_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
                upper_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
                lower_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
                lower_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
                weights = np.concatenate(
                    (lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

                upper_right_idx = high_x + high_y * self.width
                upper_left_idx = low_x + high_y * self.width
                lower_right_idx = high_x + low_y * self.width
                lower_left_idx = low_x + low_y * self.width
                row_idx = np.concatenate(
                    (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
                    int)
                valid_idx = np.squeeze(np.argwhere(is_valid))
                col_idx = np.tile(np.arange(self.height * self.width), 4)
                weight_matrix = scipy.sparse.csr_array(
                    (weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                    shape=(self.height * self.width, self.height * self.width))
                # row_sums = np.array(weight_matrix.sum(axis=1))  # Sum of each row
                # row_indices, col_indices = weight_matrix.nonzero()o
                # weight_matrix.data /= row_sums[row_indices]

                lf = self.lf[:, :, i, j].flatten()
                lf_reconstructed[:, :, i, j] = (weight_matrix @ lf).reshape(self.height, self.width)

        return lf_reconstructed
