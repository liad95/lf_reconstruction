import gc
from itertools import product

import cupy as cp
import torch
import torch.nn.functional as interpolator
from cupyx.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorGPU
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from display import *
from phase_mask_finder import phase_mask_finder
from utils import *
from debug_utils import tensor_dict, get_tensor_memory, get_gpu_memory_status


def LPF_gpu(phase_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs LPF on the phase mask angle gradient, on GPU
    :param phase_mask: the phase mask
    :return: the smoothed phase mask
    """
    ## TODO: convert this to pytorch if needed
    sigma = 1.0
    lpf_array = gaussian_filter(phase_mask.cpu().numpy(), sigma=sigma)
    return torch.from_numpy(lpf_array).to('cuda')


def LPF(phase_mask):
    """
    Performs LPF on the phase mask angle gradient
    :param phase_mask: the phase mask
    :return: the smoothed phase mask
    """
    sigma = 1.0
    lpf_array = gaussian_filter(phase_mask, sigma=sigma)
    return lpf_array


class phase_mask_finder_walker(phase_mask_finder):

    def __init__(self, n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma, phase_mask_shape,
                 lf_shape, max_sin, L, n_delta, max_delta, n_step_size, max_step_size, method):
        """
        A phase mask finder using a walker algorithm, which finds the optimal delta out of a possible list,
        with a cost function of the forward warped LF, using a gaussian mask.
        :param n_iter: number of iteration
        :param sampling_dist_mask_plane: the sampling distance of the phase mask
        :param sampling_dist_lf_plane: the sampling distance of the light field
        :param wavelength: the wavelength size
        :param sigma: the std of the gaussian mask for the cost function
        :param phase_mask_shape: the shape of the phase mask to find
        :param lf_shape: the shape of the light field
        :param max_sin: the maximum sine angle of the light field
        :param L: the distance between the light field and the phase mask layer
        :param n_delta: the number of deltas to walk over
        :param max_delta: the maximum value of delta
        :param n_step_size: the number of step sizes to try
        :param max_step_size: the maximum value of a step size
        :param method: the method of interpolation
        """
        super().__init__(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                         phase_mask_shape,
                         lf_shape, max_sin, L)

        self.n_delta = n_delta
        self.max_delta = max_delta
        self.n_step_size = n_step_size
        self.max_step_size = max_step_size
        self.method = method

    def find_phase_mask(self, lf):
        """
        Find the phase mask layer based on the recorded LF
        :param lf: the recorded light field
        :return: the phase mask angle gradient in x and y (phase_maskx, phase_masky)
        """
        for k in range(self.n_iter):
            e = self.single_iter_x_or_y(lf)
            print(f"iter #{k} - {e}")
        return self.phase_maskx, self.phase_masky

    def find_phase_mask_gpu(self, lf):
        """
        Find the phase mask layer based on the recorded LF
        :param lf: the recorded light field
        :return: the phase mask angle gradient in x and y (phase_maskx, phase_masky)
        """

        lf = self._convert_to_tensor(lf)
        for k in range(self.n_iter):
            e = self.single_iter_x_or_y_gpu(lf)
            print(f"iter #{k} - {e}")
        return self.phase_maskx, self.phase_masky

    def find_phase_mask_gpu_debug(self, lf):
        """
        Find the phase mask layer based on the recorded LF
        :param lf: the recorded light field
        :return: the phase mask angle gradient in x and y (phase_maskx, phase_masky)
        """
        self.dict = tensor_dict()
        lf = self._convert_to_tensor_debug(lf)
        # track tensors
        self.track_all()
        self.dict.track_obj(lf, "lf")
        self.dict.print_active_tensors()

        for k in range(self.n_iter):
            e = self.single_iter_x_or_y_gpu_debug(lf)
            print(f"iter #{k} - {e}")
        return self.phase_maskx, self.phase_masky

    # region old_func
    def single_iter_gpu_old(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added for each angle in the LF, on GPU
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        lf = self._convert_to_cp(lf)
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        current_energy = 0
        # create an array of possible deltas
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)

        for i in np.arange(self.lf_shape[2]):
            for j in np.arange(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                # create matrices to find delta which optimizes the score
                max_deltax = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_deltay = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorex = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorey = cp.zeros((self.lf_shape[0], self.lf_shape[1]))
                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1 + delta, angle_y1)
                    mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                           self.mask[:, :, i, j] * 1e20,
                                                           bounds_error=False, fill_value=0)
                    # The cost
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    # update the max score and max delta
                    max_deltax = cp.where(weight_x > max_scorex, delta, max_deltax)
                    max_scorex = cp.where(weight_x > max_scorex, weight_x, max_scorex)

                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1, angle_y1 + delta)
                    mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                           self.mask[:, :, i, j] * 1e20,
                                                           bounds_error=False, fill_value=0)
                    # The cost
                    weight_y = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_y = weight_y * lf[:, :, i, j]
                    # update the max score and max delta
                    max_deltay = cp.where(weight_y > max_scorey, delta, max_deltay)
                    max_scorey = cp.where(weight_y > max_scorey, weight_y, max_scorey)

                # create transformation matrix to convert to the phase_mask shape
                transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                # finding the final max delta for phase mask x and phase mask y
                gradient_x = (transform_matrix @ (max_deltax.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (max_deltay.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])

                # summing the energy (cost) of delta = 0, for this angle
                mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1)
                mask_point = cp.array([mask_x.ravel(), mask_y.ravel()]).T
                mask_func = RegularGridInterpolatorGPU((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                       self.mask[:, :, i, j],
                                                       bounds_error=False, fill_value=0)
                current_energy += cp.sum(lf[:, :, i, j] * mask_func(mask_point).reshape(mask_x.shape))
                # update the phase mask
                self._update_phase_mask_gpu(gradient_x, gradient_y)

        return current_energy

    def single_iter(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added for each angle in the LF
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        current_energy = 0

        # create an array of possible deltas
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)

                # create matrices to find delta which optimizes the score
                max_deltax = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_deltay = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorex = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                max_scorey = np.zeros((self.lf_shape[0], self.lf_shape[1]))
                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    ## TODO: Recall why I needed to multiply by 1e20
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    # The cost
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]

                    # update the max score and max delta
                    max_deltax = np.where(weight_x > max_scorex, delta, max_deltax)
                    max_scorex = np.where(weight_x > max_scorex, weight_x, max_scorex)

                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    # The cost
                    weight_y = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_y = weight_y * lf[:, :, i, j]

                    # update the max score and max delta
                    max_deltay = np.where(weight_y > max_scorey, delta, max_deltay)
                    max_scorey = np.where(weight_y > max_scorey, weight_y, max_scorey)

                # create transformation matrix to convert to the phase_mask shape
                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight_x.shape)
                # finding the final max delta for phase mask x and phase mask y
                gradient_x = (transform_matrix @ (max_deltax.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                gradient_y = (transform_matrix @ (max_deltay.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])

                # summing the energy (cost) of delta = 0, for this angle
                mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                        siny[j],
                                                        self.L,
                                                        angle_x1, angle_y1)
                mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                    self.mask[:, :, i, j],
                                                    bounds_error=False, fill_value=0)
                current_energy += np.sum(lf[:, :, i, j] * mask_func(mask_point).reshape(mask_x.shape))

                # update the phase mask
                self._update_phase_mask(gradient_x, gradient_y)

        return current_energy

    def single_iter_x_and_y(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for coupled delta_x and delta_y (not independently)
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        # create a list of possible x and y delta combinations
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        deltas = list(product(deltas, deltas))
        score = {}
        # zero scores for the different deltas
        for delta in deltas:
            score[delta] = np.zeros(self.phase_mask_shape)

        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane)

                for delta in deltas:
                    delta_x = delta[0]
                    delta_y = delta[1]
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta_x, angle_y1 + delta_y)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    # The cost
                    weight = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight = weight * lf[:, :, i, j]

                    # create transformation matrix to convert to the phase_mask shape
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight.shape)
                    # update the delta's score
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

        return np.sum(score[(0, 0)]), np.sum(np.max(score_stacked, axis=0))

    def single_iter_adv(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        Because of the way the for loops are nestled, takes a long time. Recommended to use single_iter_x_or_y
        :param lf: the given light field
        :return: the energy in the mask
        """
        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        # create delta list
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
                    phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                    # finding the gradient angle of the phase mask
                    angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane)
                    # X
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j] * 1e20,
                                                        bounds_error=False, fill_value=0)
                    # The cost
                    weight_x = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    # Y
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    # interpolation of the mask
                    mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                        self.mask[:, :, i, j],
                                                        bounds_error=False, fill_value=0)
                    # The cost
                    weight_y = mask_func(mask_point).reshape(mask_x.shape)
                    weight_y = weight_y * lf[:, :, i, j]
                    # create transformation matrix to convert to the phase_mask shape
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)
                    # update the deltas' score
                    score_x += (transform_matrix @ (weight_x.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
                    score_y += (transform_matrix @ (weight_y.flatten())).reshape(self.phase_mask_shape[0],
                                                                                 self.phase_mask_shape[1])
            # Update Max
            max_delta_x = np.where(score_x > max_score_x, delta, max_delta_x)
            max_score_x = np.where(score_x > max_score_x, score_x, max_score_x)
            max_delta_y = np.where(score_y > max_score_y, delta, max_delta_y)
            max_score_y = np.where(score_y > max_score_y, score_y, max_score_y)

        # updates the phase mask
        self._update_phase_mask(max_delta_x, max_delta_y)
        return np.max(np.sum(max_score_x), np.sum(max_score_y))

    # endregion
    def single_iter_x_or_y(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        # create a list of possible x and y delta combinations
        deltas = np.linspace(-self.max_delta, self.max_delta, self.n_delta)
        if 0 not in deltas:
            np.append(np.array([0]), deltas)
        else:
            deltas = np.concatenate((np.array([0]), np.delete(deltas, np.where(deltas == 0)[0])))
        score_x = {}
        score_y = {}
        # zero scores for the different deltas
        for delta in deltas:
            score_x[delta] = np.zeros(self.phase_mask_shape)
            score_y[delta] = np.zeros(self.phase_mask_shape)
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      self.phase_maskx, self.phase_masky,
                                                      self.sampling_dist_mask_plane, self.method)

                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1 + delta, angle_y1)
                    mask_point_x = np.array([mask_x.ravel(), mask_y.ravel()]).T
                    mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                            siny[j],
                                                            self.L,
                                                            angle_x1, angle_y1 + delta)
                    mask_point_y = np.array([mask_x.ravel(), mask_y.ravel()]).T

                    # interpolation of the mask
                    mask_func_x = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                          self.mask[:, :, i, j] * 1e20,
                                                          bounds_error=False, fill_value=0)
                    mask_func_y = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                          self.mask[:, :, i, j] * 1e20,
                                                          bounds_error=False, fill_value=0)
                    # The cost
                    weight_x = mask_func_x(mask_point_x).reshape(mask_x.shape) * 1e-20
                    weight_y = mask_func_y(mask_point_y).reshape(mask_x.shape) * 1e-20
                    weight_x = weight_x * lf[:, :, i, j]
                    weight_y = weight_y * lf[:, :, i, j]

                    # create transformation matrix to convert to the phase_mask shape
                    transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                               self.sampling_dist_mask_plane,
                                                               self.phase_mask_shape, weight_x.shape)

                    # update the deltas' score
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
        # max_score = self._update_phase_mask(lf, max_delta_x, max_delta_y)
        self._update_phase_mask_simple(lf, max_delta_x, max_delta_y)
        return max(np.sum(score_x_stacked[max_indices_x, np.arange(1601)[:, None], np.arange(1601)]),
                   np.sum(score_y_stacked[max_indices_y, np.arange(1601)[:, None], np.arange(1601)]))

    def single_iter_x_or_y_gpu(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        deltas = torch.linspace(-self.max_delta, self.max_delta, self.n_delta, device=device)
        if 0 not in deltas:
            torch.concatenate((torch.tensor([0], device=device), deltas))
        else:
            deltas = torch.concatenate(
                (torch.tensor([0], device=device), torch.delete(deltas, torch.where(deltas == 0)[0])))
        score_x = {}
        score_y = {}
        # zero scores for the different deltas
        for delta in deltas:
            score_x[float(delta)] = torch.zeros(self.phase_mask_shape, device=device)
            score_y[float(delta)] = torch.zeros(self.phase_mask_shape, device=device)
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane, self.method)

                for delta in deltas:
                    # finding the forward locations with the delta in the angle gradient
                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1 + delta, angle_y1)

                    # interpolation of the mask
                    mask_x = (mask_x / (torch.max(self.X)))
                    mask_y = (mask_y / (torch.max(self.Y)))
                    mask_loc = torch.stack((mask_x, mask_y), dim=2).unsqueeze(0)
                    mask = self.mask[:, :, i, j].unsqueeze(0).unsqueeze(0)
                    weight_x = interpolator.grid_sample(mask, mask_loc, mode=self.method,
                                                        padding_mode='zeros',
                                                        align_corners=True).squeeze()

                    mask_x, mask_y = find_forward_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                siny[j],
                                                                self.L,
                                                                angle_x1, angle_y1 + delta)
                    mask_x = (mask_x / (torch.max(self.X)))
                    mask_y = (mask_y / (torch.max(self.Y)))
                    mask_loc = torch.stack((mask_x, mask_y), dim=2).unsqueeze(0)
                    weight_y = interpolator.grid_sample(mask, mask_loc, mode=self.method,
                                                        padding_mode='zeros',
                                                        align_corners=True).squeeze()

                    # The cost

                    weight_x = weight_x * lf[:, :, i, j]
                    weight_y = weight_y * lf[:, :, i, j]

                    # create transformation matrix to convert to the phase_mask shape
                    transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                                   self.sampling_dist_mask_plane,
                                                                   self.phase_mask_shape, weight_x.shape)

                    # update the deltas' score
                    score_x[float(delta)] = (transform_matrix @ weight_x.flatten()).reshape(
                        self.phase_mask_shape[0], self.phase_mask_shape[1])
                    score_y[float(delta)] = (transform_matrix @ weight_y.flatten()).reshape(
                        self.phase_mask_shape[0], self.phase_mask_shape[1])

        # Stack the arrays along a new axis to create a 3D array
        score_x_stacked = torch.stack(list(score_x.values()), dim=0)
        score_y_stacked = torch.stack(list(score_y.values()), dim=0)

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices_x = torch.argmax(score_x_stacked, dim=0)
        max_indices_y = torch.argmax(score_y_stacked, dim=0)

        # Map the indices back to keys
        max_delta_x = torch.tensor(deltas)[max_indices_x]
        max_delta_y = torch.tensor(deltas)[max_indices_y]
        # max_score = self._update_phase_mask(lf, max_delta_x, max_delta_y)
        self._update_phase_mask_simple_gpu(lf, max_delta_x, max_delta_y)
        sum_x = torch.sum(score_x_stacked[max_indices_x, np.arange(1601)[:, None], np.arange(1601)],
                          dim=1)  # Add dim=1 to sum along desired axis
        sum_y = torch.sum(score_y_stacked[max_indices_y, np.arange(1601)[:, None], np.arange(1601)], dim=1)

        # Return the max of the two sums
        result = torch.max(sum_x, sum_y)
        return result

    def single_iter_x_or_y_gpu_debug(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # resources for the iteration
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        deltas = torch.linspace(-self.max_delta, self.max_delta, self.n_delta, device=device)
        if 0 not in deltas:
            torch.concatenate((torch.tensor([0], device=device), deltas))
        else:
            deltas = torch.concatenate(
                (torch.tensor([0], device=device), torch.delete(deltas, torch.where(deltas == 0)[0])))
        score_x = {}
        score_y = {}
        # zero scores for the different deltas
        for delta in deltas:
            """score_x[delta] = torch.zeros(self.phase_mask_shape, device=device)
            score_y[delta] = torch.zeros(self.phase_mask_shape, device=device)
            self.dict.track_obj(score_x[delta], f"score_x[{delta}]")
            self.dict.track_obj(score_y[delta], f"score_y[{delta}]")"""
            score_x[float(delta)] = torch.zeros(self.phase_mask_shape, device=device)
            score_y[float(delta)] = torch.zeros(self.phase_mask_shape, device=device)
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                           sinx[i],
                                                                           siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles_gpu(phase_mask_x, phase_mask_y,
                                                          self.phase_maskx, self.phase_masky,
                                                          self.sampling_dist_mask_plane, self.method)

                # track
                self.dict.track_obj(phase_mask_x, f'phase_mask_x,{i},{j}')
                self.dict.track_obj(phase_mask_y, f'phase_mask_y,{i},{j}')
                self.dict.track_obj(angle_x1, f'angle_x1,{i},{j}')
                self.dict.track_obj(angle_y1, f'angle_y1,{i},{j}')

                for delta in deltas:
                    score_x_tmp, score_y_tmp = self.get_delta_score(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                                    siny[j],
                                                                    angle_x1, angle_y1, delta, self.mask[:, :, i, j],
                                                                    lf[:, :, i, j], phase_mask_x,
                                                                    phase_mask_y, i, j)
                    score_x[float(delta)] += score_x_tmp
                    score_x[float(delta)] += score_y_tmp


                torch.cuda.synchronize()  # Ensure all operations have finished
                gc.collect()
                torch.cuda.empty_cache()
                self.dict.print_active_tensors()



        # Stack the arrays along a new axis to create a 3D array
        score_x_stacked = torch.stack(list(score_x.values()), dim=0)
        score_y_stacked = torch.stack(list(score_y.values()), dim=0)

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices_x = torch.argmax(score_x_stacked, dim=0)
        max_indices_y = torch.argmax(score_y_stacked, dim=0)

        # Map the indices back to keys
        max_delta_x = torch.tensor(deltas)[max_indices_x]
        max_delta_y = torch.tensor(deltas)[max_indices_y]
        # max_score = self._update_phase_mask(lf, max_delta_x, max_delta_y)
        self._update_phase_mask_simple_gpu(lf, max_delta_x, max_delta_y)
        sum_x = torch.sum(score_x_stacked[max_indices_x, np.arange(1601)[:, None], np.arange(1601)],
                          dim=1)  # Add dim=1 to sum along desired axis
        sum_y = torch.sum(score_y_stacked[max_indices_y, np.arange(1601)[:, None], np.arange(1601)], dim=1)

        # Return the max of the two sums
        result = torch.max(sum_x, sum_y)
        return result


    def _update_phase_mask_simple(self, lf, gradient_x, gradient_y):
        phase_mask_x = self.phase_maskx + gradient_x
        phase_mask_y = self.phase_masky + gradient_y
        phase_mask_x = np.where(np.abs(phase_mask_x) >= 0.5, 0.49, phase_mask_x)
        phase_mask_y = np.where(np.abs(phase_mask_y) >= 0.5, 0.49, phase_mask_y)
        phase_mask_x = LPF(phase_mask_x)
        phase_mask_y = LPF(phase_mask_y)
        self.phase_maskx = phase_mask_x
        self.phase_masky = phase_mask_y

    def _update_phase_mask_simple_gpu(self, lf, gradient_x, gradient_y):
        phase_mask_x = self.phase_maskx + gradient_x
        phase_mask_y = self.phase_masky + gradient_y
        phase_mask_x = torch.where(torch.abs(phase_mask_x) >= 0.5, 0.49, phase_mask_x)
        phase_mask_y = torch.where(torch.abs(phase_mask_y) >= 0.5, 0.49, phase_mask_y)
        phase_mask_x = LPF_gpu(phase_mask_x)
        phase_mask_y = LPF_gpu(phase_mask_y)
        self.phase_maskx = phase_mask_x
        self.phase_masky = phase_mask_y

    def _update_phase_mask(self, lf, gradient_x, gradient_y):
        """
        Updates the phase mask according to the optimal delta
        :param gradient_x: the optimal delta in the x direction
        :param gradient_y: the optimal delta in the y direction
        """
        step_sizes = np.linspace(0, self.max_step_size, self.n_step_size)
        gradient_x = LPF(gradient_x)
        gradient_y = LPF(gradient_y)
        max_score = 0
        max_phase_mask_x = self.phase_maskx
        max_phase_mask_y = self.phase_masky
        for step_size in step_sizes:
            phase_mask_x = self.phase_maskx + step_size * gradient_x
            phase_mask_y = self.phase_masky + step_size * gradient_y
            phase_mask_x = np.where(np.abs(phase_mask_x) >= 0.5, 0.49, phase_mask_x)
            phase_mask_y = np.where(np.abs(phase_mask_y) >= 0.5, 0.49, phase_mask_y)
            phase_mask_x = LPF(phase_mask_x)
            phase_mask_y = LPF(phase_mask_y)
            score = self.get_score(lf, phase_mask_x, phase_mask_y)
            print(f"step size = {step_size}, score = {score}")
            """angle_finder = gradient_angle_finder(self.sampling_dist_mask_plane, 1, self.wavelength, self.sigma)
            reconstructor = lf_forward_reconstructor(self.max_sin, self.wavelength, self.sampling_dist_lf_plane,
                                                     self.sampling_dist_mask_plane, 1,
                                                     self.L, angle_finder,
                                                     lf.shape)
            lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_mask_x, phase_mask_y)
            display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
            plt.show()"""
            if score > max_score:
                max_score = score
                max_phase_mask_x = phase_mask_x
                max_phase_mask_y = phase_mask_y

        self.phase_maskx = max_phase_mask_x
        self.phase_masky = max_phase_mask_y
        return max_score

    def _update_phase_mask_gpu(self, gradient_x, gradient_y):
        """
        Updates the phase mask according to the optimal delta, on GPU
        :param gradient_x: the optimal delta in the x direction
        :param gradient_y: the optimal delta in the y direction
        """
        gradient_x = LPF_gpu(gradient_x)
        gradient_y = LPF_gpu(gradient_y)
        self.phase_maskx = self.phase_maskx + gradient_x
        self.phase_masky = self.phase_masky + gradient_y
        self.phase_maskx = cp.where(cp.abs(self.phase_maskx) >= 0.5, 0.49, self.phase_maskx)
        self.phase_masky = cp.where(cp.abs(self.phase_masky) >= 0.5, 0.49, self.phase_masky)
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

    def get_score(self, lf, phase_maskx, phase_masky):
        sinx = self.SinX[0, 0, :, 0]
        siny = self.SinY[0, 0, 0, :]
        score = 0
        for i in range(self.lf_shape[2]):
            for j in range(self.lf_shape[2]):
                # finding the locations on the phase mask
                phase_mask_x, phase_mask_y = find_phase_mask_locations(self.X[:, :, i, j], self.Y[:, :, i, j],
                                                                       sinx[i],
                                                                       siny[j], self.L)
                # finding the gradient angle of the phase mask
                angle_x1, angle_y1 = find_mask_angles(phase_mask_x, phase_mask_y,
                                                      phase_maskx, phase_masky,
                                                      self.sampling_dist_mask_plane, self.method)

                # finding the forward locations with the delta in the angle gradient
                mask_x, mask_y = find_forward_locations(self.X[:, :, i, j], self.Y[:, :, i, j], sinx[i],
                                                        siny[j],
                                                        self.L,
                                                        angle_x1, angle_y1)
                mask_point = np.array([mask_x.ravel(), mask_y.ravel()]).T
                # interpolation of the mask
                mask_func = RegularGridInterpolator((self.X[:, 0, i, j], self.Y[0, :, i, j]),
                                                    self.mask[:, :, i, j] * 1e20,
                                                    bounds_error=False, fill_value=0)
                # The cost
                weight = mask_func(mask_point).reshape(mask_x.shape) * 1e-20
                weight = weight * lf[:, :, i, j]

                # create transformation matrix to convert to the phase_mask shape
                transform_matrix = create_transform_matrix(phase_mask_x, phase_mask_y,
                                                           self.sampling_dist_mask_plane,
                                                           self.phase_mask_shape, weight.shape)
                # update the deltas' score
                score += np.sum((transform_matrix @ (weight.flatten())).reshape(self.phase_mask_shape[0],
                                                                                self.phase_mask_shape[1]))

        return score

    def _convert_to_tensor(self, lf):
        self.mask = torch.from_numpy(self.mask).to("cuda")
        self.phase_maskx = torch.from_numpy((self.phase_maskx)).to("cuda")
        self.phase_masky = torch.from_numpy((self.phase_masky)).to("cuda")
        self.X = torch.from_numpy(self.X).to("cuda")
        self.Y = torch.from_numpy(self.Y).to("cuda")
        self.SinX = torch.from_numpy(self.SinX).to("cuda")
        self.SinY = torch.from_numpy(self.SinY).to("cuda")
        self.SinZ = torch.from_numpy(self.SinZ).to("cuda")
        self.Phase_X = torch.from_numpy(self.Phase_X).to("cuda")
        self.Phase_Y = torch.from_numpy(self.Phase_Y).to("cuda")
        return torch.from_numpy(lf).to("cuda")

    def _convert_to_tensor_debug(self, lf):
        self.mask = torch.from_numpy(self.mask).to("cuda")
        self.phase_maskx = torch.from_numpy((self.phase_maskx)).to("cuda")
        self.phase_masky = torch.from_numpy((self.phase_masky)).to("cuda")
        self.X = torch.from_numpy(self.X).to("cuda")
        self.Y = torch.from_numpy(self.Y).to("cuda")
        self.SinX = torch.from_numpy(self.SinX).to("cuda")
        self.SinY = torch.from_numpy(self.SinY).to("cuda")
        self.SinZ = torch.from_numpy(self.SinZ).to("cuda")
        self.Phase_X = torch.from_numpy(self.Phase_X).to("cuda")
        self.Phase_Y = torch.from_numpy(self.Phase_Y).to("cuda")
        return torch.from_numpy(lf).to("cuda")

    def track_all(self):
        tensor_vars = {name: value for name, value in globals().items() if isinstance(value, torch.Tensor)}
        for name, tensor in tensor_vars.items():
            self.dict.track_obj(tensor, name)

        tensor_vars = {name: value for name, value in locals().items() if isinstance(value, torch.Tensor)}
        for name, tensor in tensor_vars.items():
            self.dict.track_obj(tensor, name)

        tensor_vars = {name: value for name, value in vars(self).items() if isinstance(value, torch.Tensor)}
        for name, tensor in tensor_vars.items():
            self.dict.track_obj(tensor, name)

    def get_delta_score(self, X, Y, sinx, siny, angle_x1, angle_y1, delta, mask, lf, phase_mask_x, phase_mask_y, i, j):
        # finding the forward locations with the delta in the angle gradient
        mask_x, mask_y = find_forward_locations_gpu(X, Y, sinx, siny, self.L, angle_x1 + delta, angle_y1)

        # interpolation of the mask
        mask_x = (mask_x / (torch.max(X)))
        mask_y = (mask_y / (torch.max(Y)))
        mask_loc = torch.stack((mask_x, mask_y), dim=2).unsqueeze(0)

        mask = mask.unsqueeze(0).unsqueeze(0)
        weight_x = interpolator.grid_sample(mask, mask_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()

        mask_x, mask_y = find_forward_locations_gpu(X, Y, sinx, siny, self.L, angle_x1, angle_y1 + delta)
        mask_x = (mask_x / (torch.max(X)))
        mask_y = (mask_y / (torch.max(Y)))
        mask_loc = torch.stack((mask_x, mask_y), dim=2).unsqueeze(0)
        weight_y = interpolator.grid_sample(mask, mask_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()

        # The cost

        weight_x = weight_x * lf
        weight_y = weight_y * lf

        # create transformation matrix to convert to the phase_mask shape
        transform_matrix = create_transform_matrix_gpu(phase_mask_x, phase_mask_y,
                                                       self.sampling_dist_mask_plane,
                                                       self.phase_mask_shape, weight_x.shape)

        # update the deltas' score
        score_x = (transform_matrix @ weight_x.flatten()).reshape(
            self.phase_mask_shape[0], self.phase_mask_shape[1])
        score_y = (transform_matrix @ weight_y.flatten()).reshape(
            self.phase_mask_shape[0], self.phase_mask_shape[1])

        self.dict.track_obj(mask_x, f'mask_x,{i},{j},{delta}')
        self.dict.track_obj(mask_y, f'mask_y,{i},{j},{delta}')
        self.dict.track_obj(mask, f'mask_tmp,{i},{j},{delta}')
        self.dict.track_obj(mask_loc, f'mask_loc,{i},{j},{delta}')
        self.dict.track_obj(weight_x, f'weight_x,{i},{j},{delta}')
        self.dict.track_obj(weight_y, f'weight_y,{i},{j},{delta}')
        self.dict.track_obj(transform_matrix, f'transform_matrix,{i},{j},{delta}')
        del mask_x
        del mask_y
        del mask
        del mask_loc
        del weight_x
        del weight_y
        del transform_matrix

        return score_x, score_y
