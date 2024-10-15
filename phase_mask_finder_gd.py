import numpy as np
import scipy
from scipy.sparse import linalg
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from lf_reconstructor import lf_reconstructor
import math
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorGPU
from joblib import Parallel, delayed
from itertools import product
import dask
from dask import delayed
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from phase_mask_finder import phase_mask_finder
from scipy.ndimage import gaussian_filter
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filterGPU


def LPF_gpu(phase_mask):
    """
    Performs LPF on the phase mask angle gradient, on GPU
    :param phase_mask: the phase mask
    :return: the smoothed phase mask
    """
    sigma = 1.0
    lpf_array = gaussian_filterGPU(phase_mask, sigma=sigma)
    return lpf_array


def LPF(phase_mask):
    """
    Performs LPF on the phase mask angle gradient
    :param phase_mask: the phase mask
    :return: the smoothed phase mask
    """
    sigma = 1.0
    lpf_array = gaussian_filter(phase_mask, sigma=sigma)
    return lpf_array


class phase_mask_finder_gd(phase_mask_finder):

    def __init__(self, n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma, phase_mask_shape,
                 lf_shape, max_sin, L, step_size, delta):
        """
        A phase mask finder using a gradient descent algorithm, with a cost function of the forward warped LF,
        using a gaussian mask.
        :param n_iter: number of iteration
        :param sampling_dist_mask_plane: the sampling distance of the phase mask
        :param sampling_dist_lf_plane: the sampling distance of the light field
        :param wavelength: the wavelength size
        :param sigma: the std of the gaussian mask for the cost function
        :param phase_mask_shape: the shape of the phase mask to find
        :param lf_shape: the shape of the light field
        :param max_sin: the maximum sine angle of the light field
        :param L: the distance between the light field and the phase mask layer
        :param step_size: the step size for the gradient descent update
        :param delta: the delta for the gradient calculation
        """
        super().__init__(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                         phase_mask_shape,
                         lf_shape, max_sin, L)

        self.delta = delta
        self.step_size = step_size

    def find_phase_mask(self, lf):
        """
        Find the phase mask layer based on the recorded LF
        :param lf: the recorded light field
        :return: the phase mask angle gradient in x and y (phase_maskx, phase_masky)
        """
        energy = np.array([])
        for k in range(self.n_iter):
            energy = np.append(energy, self.single_iter(lf))
            print(f"iter #{k} with energy of {energy}")
        return self.phase_maskx, self.phase_masky

    def single_iter_gpu(self, lf):
        """
        A single iteration of the gradient descent to maximize the energy of the warped light field in the defined mask, on GPU
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        lf = self._convert_to_cp(lf)
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = cp.zeros(self.phase_mask_shape)
        gradient_y = cp.zeros(self.phase_mask_shape)
        current_energy = 0

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                # finding the forward locations with and without delta in the angle gradient
                mask_x1, mask_y1 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1, angle_y1)
                mask_x2, mask_y2 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1 + self.delta, angle_y1)
                mask_x3, mask_y3 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1, angle_y1 + self.delta)
                mask_point1 = cp.array([mask_x1.ravel(), mask_y1.ravel()]).T
                mask_point2 = cp.array([mask_x2.ravel(), mask_y2.ravel()]).T
                mask_point3 = cp.array([mask_x3.ravel(), mask_y3.ravel()]).T
                # interpolation of the mask
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                # The cost difference of the deltas, with normalization by the delta size
                weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = weight_x * lf[:, :, i, j] / self.delta
                weight_y = weight_y * lf[:, :, i, j] / self.delta
                # create transformation matrix to convert to the phase_mask shape
                transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                # summing the gradient weight of this angle in the LF
                gradient_x = (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])
                # summing cost
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))
        # updates the phase mask
        self._update_phase_mask_gpu(gradient_x, gradient_y)
        return current_energy

    def single_iter_gpu2(self, lf):
        """
        A single iteration of the gradient descent to maximize the energy of the warped light field in the defined
        mask, on GPU. The transformation matrix is built as a sparse block matrix(the transformation is performed at
        once for all angles), with fewer steps.
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        lf = self._convert_to_cp(lf)
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = cp.zeros(self.phase_mask_shape)
        gradient_y = cp.zeros(self.phase_mask_shape)
        current_energy = 0;
        weight_x = cp.array([])
        weight_y = cp.array([])
        weights = cp.array([])
        row_idx = cp.array([])
        col_idx = cp.array([])

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                # finding the forward locations with and without delta in the angle gradient
                mask_x1, mask_y1 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1, angle_y1)
                mask_x2, mask_y2 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1 + self.delta, angle_y1)
                mask_x3, mask_y3 = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                              self.L,
                                                              angle_x1, angle_y1 + self.delta)
                mask_point1 = cp.array([mask_x1.ravel(), mask_y1.ravel()]).T
                mask_point2 = cp.array([mask_x2.ravel(), mask_y2.ravel()]).T
                mask_point3 = cp.array([mask_x3.ravel(), mask_y3.ravel()]).T
                # interpolation of the mask
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                # The cost difference of the deltas, with normalization by the delta size
                weight_x_tmp = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y_tmp = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = cp.append(weight_x, weight_x_tmp * lf[:, :, i, j] / self.delta)
                weight_y = cp.append(weight_y, weight_y_tmp * lf[:, :, i, j] / self.delta)
                # find the weights, row and cols for the transformation matrix
                weights_tmp, row_idx_tmp, col_idx_tmp = create_transform_matrix_gpu_weight_row_col(phase_mask_x,
                                                                                                   phase_mask_y,
                                                                                                   self.sampling_dist_mask_plane,
                                                                                                   self.phase_mask_shape,
                                                                                                   weight_x_tmp.shape)
                # append the weights, row and cols for a large block sparse transformation matrix
                weights = cp.append(weights, weights_tmp)
                row_idx = cp.append(row_idx, row_idx_tmp)
                col_idx = cp.append(col_idx, col_idx_tmp)
                # summing cost
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))
        # creating the large block sparse transformation matrix
        transform_matrix = csr_gpu((weights, (row_idx, col_idx)),
                                   shape=(self.phase_mask_shape[0] * self.phase_mask_shape[1],
                                          self.lf_shape[0] * self.lf_shape[1] * self.lf_shape[2] * self.lf_shape[2]))
        # summing the gradient weight of this angle in the LF
        gradient_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                        self.phase_mask_shape[1])
        gradient_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                        self.phase_mask_shape[1])
        # updates the phase mask
        self._update_phase_mask_gpu(gradient_x, gradient_y)
        return current_energy

    def single_iter(self, lf):
        """
        A single iteration of the gradient descent to maximize the energy of the warped light field in the defined mask.
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        norm_factor = np.sum(lf)
        current_energy = 0

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                       siny[j], self.L)

                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)

                # finding the forward locations with and without delta in the angle gradient
                mask_x1, mask_y1 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1, angle_y1)
                mask_x2, mask_y2 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1 + self.delta, angle_y1)
                mask_x3, mask_y3 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1, angle_y1 + self.delta)

                mask_point1 = np.array([mask_x1.ravel(), mask_y1.ravel()]).T
                mask_point2 = np.array([mask_x2.ravel(), mask_y2.ravel()]).T
                mask_point3 = np.array([mask_x3.ravel(), mask_y3.ravel()]).T
                # interpolation of the mask
                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                    bounds_error=False, fill_value=0)

                # The cost difference of the deltas, with normalization by the delta size, and norm_factor
                weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = weight_x * lf[:, :, i, j] / self.delta
                weight_y = weight_y * lf[:, :, i, j] / self.delta
                weight_x = weight_x / norm_factor
                weight_y = weight_y / norm_factor

                # create transformation matrix to convert to the phase_mask shape
                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight_x.shape)

                # summing the gradient weight of this angle in the LF
                gradient_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                                self.phase_mask_shape[1])
                gradient_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                                self.phase_mask_shape[1])

                # summing cost
                current_energy += np.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))

        # updates the phase mask
        self._update_phase_mask(gradient_x, gradient_y)
        return current_energy

    def _update_phase_mask(self, gradient_x, gradient_y):
        """
        Updates the phase mask according to the calculated gradients
        :param gradient_x: the gradient in the x direction
        :param gradient_y: the gradient in the y direction
        """
        # perform the gradient descent step
        self.phase_maskx = self.phase_maskx + self.step_size_func(gradient_x) * gradient_x
        self.phase_masky = self.phase_masky + self.step_size_func(gradient_y) * gradient_y
        # make sure the angle is not above sin=0.5
        self.phase_maskx = np.where(np.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = np.where(np.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)
        # perform LPF over the phase mask angle gradient
        self.phase_maskx = LPF(self.phase_maskx)
        self.phase_masky = LPF(self.phase_masky)

    def _update_phase_mask_gpu(self, gradient_x, gradient_y):
        """
        Updates the phase mask according to the calculated gradients, on GPU
        :param gradient_x: the gradient in the x direction
        :param gradient_y: the gradient in the y direction
        """
        # perform the gradient descent step
        self.phase_maskx = self.phase_maskx + gradient_x
        self.phase_masky = self.phase_masky + gradient_y
        # make sure the angle is not above sin=0.5
        self.phase_maskx = np.where(np.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = np.where(np.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)
        # perform LPF over the phase mask angle gradient
        self.phase_maskx = LPF_gpu(self.phase_maskx)
        self.phase_masky = LPF_gpu(self.phase_masky)

    def _convert_to_cp(self, lf):
        """
        Converts properties from numpy to cupy
        :param lf: the light field
        :return: returns the light field on gpu
        """
        self.mask = cp.array(self.mask)
        self.phase_maskx = cp.array(self.phase_maskx)
        self.phase_masky = cp.array(self.phase_masky)
        self.X = cp.array(self.X)
        self.Y = cp.array(self.Y)
        self.SinX = cp.array(self.SinX)
        self.SinY = cp.array(self.SinY)
        self.SinZ = cp.array(self.SinZ)
        self.Phase_X = cp.array(self.Phase_X)
        self.Phase_Y = cp.array(self.Phase_Y)
        return cp.array(lf)

    def single_iter_test(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        self.phase_maskx = np.ones(self.phase_mask_shape) * (-1 / 6)
        self.phase_masky = np.ones(self.phase_mask_shape) * (-1 / 6)
        i = 4
        j = 4
        # finding the locations on the phase mask
        # we can define once the interpolation points interp1
        phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                               siny[j], self.L)
        angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                              self.phase_maskx, self.phase_masky,
                                              self.sampling_dist_mask_plane)
        mask_x1, mask_y1 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                  self.L,
                                                  angle_x1, angle_y1)

        self.phase_maskx[798:802, 798:802] = 0
        self.phase_masky[798:802, 798:802] = 0
        # self.phase_maskx[800, 800] = 0
        # self.phase_masky[800, 800] = 0
        angle_x2, angle_y2 = find_mask_angles(phase_mask_x, phase_mask_y,
                                              self.phase_maskx, self.phase_masky,
                                              self.sampling_dist_mask_plane)
        mask_x2, mask_y2 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                  self.L,
                                                  angle_x2, angle_y2)
        mask_point1 = np.array([mask_x1.ravel(), mask_y1.ravel()]).T
        mask_point2 = np.array([mask_x2.ravel(), mask_y2.ravel()]).T
        mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                            bounds_error=False, fill_value=0)
        weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
            mask_x1.shape)
        display(weight_x, "weightx")
        plt.show()
        transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                   self.sampling_dist_mask_plane,
                                                   self.phase_mask_shape, weight_x.shape)
        gradient_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                        self.phase_mask_shape[1])
        display(gradient_x, "gradientx")
        plt.show()
        pass

    def single_iter_test2(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        self.phase_maskx = np.ones(self.phase_mask_shape) * (-1 / 6)
        self.phase_masky = np.ones(self.phase_mask_shape) * (-1 / 6)
        self.mask = np.ones_like(self.mask)
        for i in np.arange(7):
            for j in np.arange(7):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                       siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)
                mask_x1, mask_y1 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1, angle_y1)
                mask_point1 = np.array([mask_x1.ravel(), mask_y1.ravel()]).T

                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                    bounds_error=False, fill_value=0)
                # weight_x = mask_func(mask_point1).reshape(mask_x1.shape)
                weight_x = np.ones((241, 241))
                # display(weight_x, "weightx")
                # plt.show()
                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight_x.shape)
                grad = (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                         self.phase_mask_shape[1])
                # display(grad, f"i={i}, j={j}")
                gradient_x += grad
        display(gradient_x, "gradient")
        plt.show()
        pass

    def step_size_func(self, gradient):
        ## TODO: Should this be a sigmoid??? shouldn't it be a monotonously decreasing function?
        """
        Returns a dynamic step size for the gradient descent step, which depends on the gradient.
        We are using a sigmoid function
        :param gradient: the gradient
        :return: the step size
        """
        return self.step_size * 2 / (1 + np.exp(-gradient))
