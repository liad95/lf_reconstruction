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
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filterGPU
from joblib import Parallel, delayed
from itertools import product
import dask
from dask import delayed
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from phase_mask_finder import phase_mask_finder
from scipy.ndimage import gaussian_filter


class phase_mask_finder_walker(phase_mask_finder):

    def __init__(self, n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma, phase_mask_shape,
                 lf_shape, max_sin, L, n_delta, max_delta, method):
        super().__init__(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma,
                         phase_mask_shape,
                         lf_shape, max_sin, L)

        self.n_delta = n_delta
        self.max_delta = max_delta
        self.method = method

    def find_phase_mask(self, lf):
        energy0 = np.array([])
        energy1 = np.array([])
        print(np.sum(lf))
        for k in range(self.n_iter):
            #e0, e1 = self.single_iter_adv2(lf)
            e = self.single_iter_x_or_y(lf)
            #energy0 = np.append(energy0, e0)
            #energy1 = np.append(energy1, e1)
            print(f"iter #{k}")
        #plt.figure()
        """plt.plot(energy0)
        plt.plot(energy1)
        plt.show()"""
        return self.phase_maskx, self.phase_masky

    def single_iter_gpu(self, lf):
        lf = self._convert_to_cp(lf)
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = cp.zeros(self.phase_mask_shape)
        gradient_y = cp.zeros(self.phase_mask_shape)
        current_energy = 0
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)

        """for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):"""
        for i in np.arange(3, self.lf_shape[2]):
            for j in np.arange(3, self.lf_shape[2]):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                max_deltax = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_deltay = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorex = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorey = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                delta_scoresx = {}
                delta_scoresy = {}
                for delta in deltas:
                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1 + delta, angle_y1)
                    mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                           self.mask[:, :, i, j] * 1e20,
                                                           bounds_error=False, fill_value=0)
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    plt.show()
                    # display(cp.asnumpy(weight_x), f"Display of Weight X for i={i}, j={j} with Sum = {cp.sum(weight_x)}")
                    delta_scoresx[delta] = cp.sum(weight_x)
                    max_deltax = cp.where(weight_x > max_scorex, delta, max_deltax)
                    max_scorex = cp.where(weight_x > max_scorex, weight_x, max_scorex)

                for delta in deltas:
                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1, angle_y1 + delta)
                    mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                           self.mask[:, :, i, j],
                                                           bounds_error=False, fill_value=0)
                    weight_y = mask_func(mask_point).reshape(mask_x.shape)
                    weight_y = weight_y * lf[:, :, i, j]
                    # display(cp.asnumpy(weight_y),f"Display of Weight Y for i={i}, j={j} with Sum = {cp.sum(weight_y)}")
                    plt.show()
                    max_deltay = cp.where(weight_y > max_scorey, delta, max_deltay)
                    max_scorey = cp.where(weight_y > max_scorey, weight_y, max_scorey)

                transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                gradient_x = (transform_matrix @ (max_deltax.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (max_deltay.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1)
                mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                       self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point).reshape(mask_x.shape))
                self._update_phase_mask_gpu(gradient_x, gradient_y)

        return current_energy

    def single_iter(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        current_energy = 0
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # for i in np.arange(3, 4):
                # for j in np.arange(3, 4):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)
                max_deltax = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_deltay = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorex = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorey = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                delta_scoresx = {}
                delta_scoresy = {}
                for delta in deltas:
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    # plt.show()
                    # display(np.asnumpy(weight_x), f"Display of Weight X for i={i}, j={j} with Sum = {np.sum(weight_x)}")
                    delta_scoresx[delta] = np.sum(weight_x)
                    max_deltax = np.where(weight_x > max_scorex, delta, max_deltax)
                    max_scorex = np.where(weight_x > max_scorex, weight_x, max_scorex)

                for delta in deltas:
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j],
                                                        bounds_error=False, fill_value=0)
                    weight_y = mask_func(mask_point).reshape(mask_x.shape)
                    weight_y = weight_y * lf[:, :, i, j]
                    # display(np.asnumpy(weight_y),f"Display of Weight Y for i={i}, j={j} with Sum = {cp.sum(weight_y)}")
                    plt.show()
                    max_deltay = np.where(weight_y > max_scorey, delta, max_deltay)
                    max_scorey = np.where(weight_y > max_scorey, weight_y, max_scorey)

                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight_x.shape)
                gradient_x = (transform_matrix @ (max_deltax.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (max_deltay.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                        siny[j],
                                                        self.L,
                                                        angle_x1, angle_y1)
                mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                    self.mask[:, :, i, j],
                                                    bounds_error=False, fill_value=0)
                current_energy += np.sum(lf[:, :, i, j] * mask_func(mask_point).reshape(mask_x.shape))
                self._update_phase_mask(gradient_x, gradient_y)

        return current_energy

    def single_iter_x_and_y(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        current_energy = 0
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        deltas = list(product(deltas, deltas))
        max_score = np.zeros(self.phase_mask_shape)
        max_delta = np.zeros(self.phase_mask_shape)
        score = {}
        for delta in deltas:
            score[delta] = np.zeros(self.phase_mask_shape)
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)
                # X
                for delta in deltas:
                    delta_x = delta[0]
                    delta_y = delta[1]
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta_x, angle_y1 + delta_y)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    weight = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight = weight * lf[:, :, i, j]
                    # Y
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight.shape)
                    score[delta] += (transform_matrix @ (weight.flatten())).reshape(self.phase_mask_shape[0],
                                                                                    self.phase_mask_shape[1])

        # Stack the arrays along a new axis to create a 3D array
        score_stacked = np.stack(np.array(list(score.values())), axis=0)

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices = np.argmax(score_stacked, axis=0)

        # Map the indices back to keys
        max_delta = np.array(deltas)[max_indices]
        max_delta_x = max_delta[:, :, 0]
        max_delta_y = max_delta[:, :, 1]
        self._update_phase_mask(max_delta_x, max_delta_y)

        return np.sum(score[(0,0)]),np.sum(np.max(score_stacked, axis=0))

    def single_iter_x_or_y(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        current_energy = 0
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        max_score_x = np.zeros(self.phase_mask_shape)
        max_delta_x = np.zeros(self.phase_mask_shape)
        max_score_y = np.zeros(self.phase_mask_shape)
        max_delta_y = np.zeros(self.phase_mask_shape)
        score_x = {}
        score_y = {}
        for delta in deltas:
            score_x[delta] = np.zeros(self.phase_mask_shape)
            score_y[delta] = np.zeros(self.phase_mask_shape)
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                # we can define once the interpolation points interp1
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane, self.method)
                # X
                for delta in deltas:
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    # Y
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j],
                                                        bounds_error=False, fill_value=0)
                    weight_y = mask_func(mask_point).reshape(mask_x.shape)
                    weight_y = weight_y * lf[:, :, i, j]
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                    score_x[delta] += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                                        self.phase_mask_shape[1])
                    score_y[delta] += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                                        self.phase_mask_shape[1])

        # Stack the arrays along a new axis to create a 3D array
        score_x_stacked = np.stack(np.array(list(score_x.values())), axis=0)
        score_y_stacked = np.stack(np.array(list(score_y.values())), axis=0)

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices_x = np.argmax(score_x_stacked, axis=0)
        max_indices_y = np.argmax(score_y_stacked, axis=0)

        # Map the indices back to keys
        max_delta_x = np.array(deltas)[max_indices_x]
        max_delta_y = np.array(deltas)[max_indices_y]
        self._update_phase_mask(max_delta_x, max_delta_y)

        sums_x = np.sum(score_x_stacked, axis=(1, 2))
        sums_y = np.sum(score_y_stacked, axis=(1, 2))
        return max(np.max(sums_x), np.max(sums_y))

    def single_iter_adv(self, lf):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        gradient_x = np.zeros(self.phase_mask_shape)
        gradient_y = np.zeros(self.phase_mask_shape)
        current_energy = 0
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        max_score_x = np.zeros(self.phase_mask_shape)
        max_delta_x = np.zeros(self.phase_mask_shape)
        max_score_y = np.zeros(self.phase_mask_shape)
        max_delta_y = np.zeros(self.phase_mask_shape)
        for delta in deltas:
            score_x = np.zeros(self.phase_mask_shape)
            score_y = np.zeros(self.phase_mask_shape)
            for i in range(self.lf_shape[2]):
                for j in range(self.lf_shape[2]):
                    # finding the locations on the phase mask
                    # we can define once the interpolation points interp1
                    phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                    angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                    # X
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    # Y
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j],
                                                        bounds_error=False, fill_value=0)
                    weight_y = mask_func(mask_point).reshape(mask_x.shape)
                    weight_y = weight_y * lf[:, :, i, j]
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                    score_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                    score_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
            # Update Max
            max_delta_x = np.where(score_x > max_score_x, delta, max_delta_x)
            max_score_x = np.where(score_x > max_score_x, score_x, max_score_x)
            max_delta_y = np.where(score_y > max_score_y, delta, max_delta_y)
            max_score_y = np.where(score_y > max_score_y, score_y, max_score_y)

        self._update_phase_mask(max_delta_x, max_delta_y)
        return np.max(np.sum(max_score_x), np.sum(max_score_y))

    def _update_phase_mask_parallel(self):
        maximum_gradient_x = np.max(self.gradient_x)
        maximum_gradient_y = np.max(self.gradient_y)
        self.phase_maskx += self.step_size / maximum_gradient_x * self.gradient_x
        self.phase_masky += self.step_size / maximum_gradient_y * self.gradient_y
        print(f"gradient x = {np.mean(self.gradient_x)}")
        print(f"gradient y = {np.mean(self.gradient_y)}")

    def _update_phase_mask(self, gradient_x, gradient_y):
        gradient_x = self.LPF(gradient_x)
        gradient_y = self.LPF(gradient_y)
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        


        self.phase_maskx = self.phase_maskx
        self.phase_masky = self.phase_masky + self.LPF(gradient_y)
        self.phase_maskx = np.where(np.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = np.where(np.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)


    def _update_phase_mask_gpu(self, gradient_x, gradient_y):
        self.phase_maskx = self.LPF_gpu(self.phase_maskx + gradient_x)
        self.phase_masky = self.LPF_gpu(self.phase_masky + gradient_y)
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
