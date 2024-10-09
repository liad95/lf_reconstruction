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


class phase_mask_finder_gd(phase_mask_finder):

    def __init__(self, n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma, phase_mask_shape,
                 lf_shape, max_sin, L, step_size, delta):
        super().__init__(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma,
                         phase_mask_shape,
                         lf_shape, max_sin, L)

        self.delta = delta
        self.step_size = step_size
        #self.phase_maskx = -1 / 6 * np.ones(self.phase_mask_shape)
        #self.phase_masky = -1 / 6 * np.ones(self.phase_mask_shape)

    def find_phase_mask(self, lf):
        print(np.sum(lf))
        print(np.std(self.mask))
        print(np.sum(lf) / np.std(self.mask))
        energy = np.array([])
        for k in range(self.n_iter):
            energy = np.append(energy, self.single_iter(lf))
            print(f"iter #{k} with energy of {energy}")
        plt.figure()
        plt.plot(energy)
        plt.show()
        return self.phase_maskx, self.phase_masky

    def single_iter_gpu(self, lf):
        lf = self._convert_to_cp(lf)
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = cp.zeros(self.phase_mask_shape)
        gradient_y = cp.zeros(self.phase_mask_shape)
        current_energy = 0

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
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
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = weight_x * lf[:, :, i, j] / self.delta
                weight_y = weight_y * lf[:, :, i, j] / self.delta
                transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                gradient_x = (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))
        self._update_phase_mask_gpu(gradient_x, gradient_y)
        return current_energy

    def single_iter_gpu2(self, lf):
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
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
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
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                weight_x_tmp = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y_tmp = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = cp.append(weight_x, weight_x_tmp * lf[:, :, i, j] / self.delta)
                weight_y = cp.append(weight_y, weight_y_tmp * lf[:, :, i, j] / self.delta)
                weights_tmp, row_idx_tmp, col_idx_tmp = create_transform_matrix_gpu2(phase_mask_x, phase_mask_y,
                                                                                     self.sampling_dist_mask_plane,
                                                                                     self.phase_mask_shape,
                                                                                     weight_x_tmp.shape)
                weights = cp.append(weights, weights_tmp)
                row_idx = cp.append(row_idx, row_idx_tmp)
                col_idx = cp.append(col_idx, col_idx_tmp)
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))

        transform_matrix = csr_gpu((weights, (row_idx, col_idx)),
                                   shape=(self.phase_mask_shape[0] * self.phase_mask_shape[1],
                                          self.lf_shape[0] * self.lf_shape[1] * self.lf_shape[2] * self.lf_shape[2]))
        gradient_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                        self.phase_mask_shape[1])
        gradient_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                        self.phase_mask_shape[1])
        self._update_phase_mask_gpu(gradient_x, gradient_y)
        return current_energy

    def process_task(self, indexes):
        i = indexes[0]
        j = indexes[1]
        # finding the locations on the phase mask
        # we can define once the interpolation points interp1
        phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j], self.sinx[i],
                                                               self.siny[j], self.L)
        angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                              self.phase_maskx, self.phase_masky,
                                              self.sampling_dist_mask_plane)
        mask_x1, mask_y1 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], self.sinx[i], self.siny[j],
                                                  self.L,
                                                  angle_x1, angle_y1)
        mask_x2, mask_y2 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], self.sinx[i], self.siny[j],
                                                  self.L,
                                                  angle_x1 + self.delta, angle_y1)
        mask_x3, mask_y3 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], self.sinx[i], self.siny[j],
                                                  self.L,
                                                  angle_x1, angle_y1 + self.delta)
        mask_point1 = np.array([mask_x1.ravel(), mask_y1.ravel()]).T
        mask_point2 = np.array([mask_x2.ravel(), mask_y2.ravel()]).T
        mask_point3 = np.array([mask_x3.ravel(), mask_y3.ravel()]).T
        mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                            bounds_error=False, fill_value=0)
        weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
            mask_x1.shape)
        weight_y = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
            mask_x1.shape)
        weight_x = weight_x * self.lf[:, :, i, j] / self.delta
        weight_y = weight_y * self.lf[:, :, i, j] / self.delta
        transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                   self.sampling_dist_mask_plane,
                                                   self.phase_mask_shape, weight_x.shape)
        gradient_x = (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                       self.phase_mask_shape[1])
        gradient_y = (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                       self.phase_mask_shape[1])
        current_energy = np.sum(self.lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))
        return gradient_x, gradient_y, current_energy

    def single_iter_parallel(self, lf):
        self.sinx = self.SinX[0, 0, :, 0]
        self.siny = self.SinY[0, 0, 0, :]
        self.lf = lf
        indexes = list(product(np.arange(self.lf_shape[2]), np.arange(self.lf_shape[2])))
        results = dask.compute(*(self.process_task(index) for index in indexes))
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        current_energy = 0
        for result in results:
            gradient_x += result[0]
            gradient_y += result[1]
            current_energy += result[2]
        self._update_phase_mask(gradient_x, gradient_y)
        return current_energy

    def single_iter(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        norm_factor = np.sum(lf)
        current_energy = 0
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
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
                mask_x2, mask_y2 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1 + self.delta, angle_y1)
                mask_x3, mask_y3 = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i], siny[j],
                                                          self.L,
                                                          angle_x1, angle_y1 + self.delta)
                mask_point1 = np.array([mask_x1.ravel(), mask_y1.ravel()]).T
                mask_point2 = np.array([mask_x2.ravel(), mask_y2.ravel()]).T
                mask_point3 = np.array([mask_x3.ravel(), mask_y3.ravel()]).T
                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]), self.mask[:, :, i, j],
                                                    bounds_error=False, fill_value=0)
                weight_x = mask_func(mask_point2).reshape(mask_x2.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_y = mask_func(mask_point3).reshape(mask_x3.shape) - mask_func(mask_point1).reshape(
                    mask_x1.shape)
                weight_x = weight_x * lf[:, :, i, j] / self.delta
                weight_y = weight_y * lf[:, :, i, j] / self.delta
                weight_x = weight_x / norm_factor
                weight_y = weight_y / norm_factor
                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight_x.shape)
                gradient_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                                self.phase_mask_shape[1])
                gradient_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                                self.phase_mask_shape[1])
                current_energy += np.sum(lf[:, :, i, j] * mask_func(mask_point1).reshape(mask_x1.shape))
        #display(gradient_x, "Gradient X")
        #display(gradient_y, "Gradient y")
        plt.show()
        self._update_phase_mask(gradient_x, gradient_y)
        return current_energy

    def _update_phase_mask_parallel(self):
        maximum_gradient_x = np.max(self.gradient_x)
        maximum_gradient_y = np.max(self.gradient_y)
        self.phase_maskx += self.step_size / maximum_gradient_x * self.gradient_x
        self.phase_masky += self.step_size / maximum_gradient_y * self.gradient_y
        print(f"gradient x = {np.max(self.gradient_x)}")
        print(f"gradient y = {np.max(self.gradient_y)}")

    def _update_phase_mask(self, gradient_x, gradient_y):
        self.phase_maskx = self.phase_maskx + self.step_size_func(gradient_x) * gradient_x
        self.phase_masky = self.phase_masky + self.step_size_func(gradient_y) * gradient_y
        self.phase_maskx = np.where(np.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = np.where(np.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)
        self.phase_maskx = self.LPF(self.phase_maskx)
        self.phase_masky = self.LPF(self.phase_masky)
        # display(self.phase_maskx+1/6, "PhaseX")
        # plt.show()
        # display(self.phase_maskx, "X")
        # display(self.phase_masky, "Y")
        """display(cp.asnumpy(self.phase_maskx), "X")
        display(cp.asnumpy(self.phase_masky), "Y")"""
        # print(f"mean phase x = {np.mean(self.phase_maskx)}")
        # print(f"mean phase y = {np.mean(self.phase_masky)}")
        # plt.show()

    def _update_phase_mask_gpu(self, gradient_x, gradient_y):
        self.phase_maskx = self.phase_maskx + gradient_x
        self.phase_masky = self.phase_masky + gradient_y
        self.phase_maskx = np.where(np.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = np.where(np.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)
        self.phase_maskx = self.LPF_gpu(self.phase_maskx)
        self.phase_masky = self.LPF_gpu(self.phase_masky)
        """display(cp.asnumpy(self.phase_maskx), "X")
        display(cp.asnumpy(self.phase_masky), "Y")"""
        # print(f"mean phase x = {cp.mean(self.phase_maskx)}")
        # print(f"mean phase y = {cp.mean(self.phase_masky)}")
        # plt.show()

    def _convert_to_cp(self, lf):
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

    def LPF_gpu(self, max_delta):
        sigma = 1.0
        lpf_array = gaussian_filterGPU(max_delta, sigma=sigma)
        return lpf_array

    def LPF(self, max_delta):
        sigma = 1.0
        lpf_array = gaussian_filter(max_delta, sigma=sigma)
        return lpf_array

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
        #self.phase_maskx[800, 800] = 0
        #self.phase_masky[800, 800] = 0
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
                #display(grad, f"i={i}, j={j}")
                gradient_x += grad
        display(gradient_x, "gradient")
        plt.show()
        pass

    def step_size_func(self, gradient):
        return self.step_size * 2 / (1 + np.exp(-gradient))
