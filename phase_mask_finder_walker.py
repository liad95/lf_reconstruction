import torch.nn.functional as F
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
import time
import matplotlib.pyplot as plt


def LPF(phase_mask, sigma):
    """
    Performs LPF on the phase mask angle gradient
    :param phase_mask: the phase mask
    :return: the smoothed phase mask
    """
    # Parameters
    kernel_size = max(int(5 * sigma), 1)
    kernel_2d = gaussian_kernel2d(kernel_size, sigma).cuda()

    output_tensor = F.conv2d(phase_mask.unsqueeze(0), kernel_2d, padding='same').squeeze()
    return output_tensor


def gaussian_kernel2d(kernel_size, sigma):
    # Create a 2D Gaussian kernel
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_2d = torch.outer(gaussian_1d, gaussian_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, kernel_size, kernel_size)  # Shape for 2D convolution


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

    def find_phase_mask_gpu(self, lf):
        """
        Find the phase mask layer based on the recorded LF
        :param lf: the recorded light field
        :return: the phase mask angle gradient in x and y (phase_maskx, phase_masky)
        """
        lf = self._convert_to_tensor(lf)
        e = []
        for k in range(self.n_iter):
            start_time = time.time()  # Record the start time
            e.append(self.single_iter_x_and_y_gpu_parallel2(lf).cpu().numpy())
            print(f"iter #{k} - {e}")
            end_time = time.time()  # Record the end time
            iteration_time = end_time - start_time  # Calculate the time taken
            print(f"Iteration {k + 1} took {iteration_time:.4f} seconds")

        plt.figure()
        plt.plot(e)
        plt.show()
        return self.phase_maskx.cpu().numpy(), self.phase_masky.cpu().numpy()

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


    def _update_phase_mask_simple_gpu_parallel2(self, lf, max_delta_x, max_delta_y):
        ## Does not work
        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        step_sizes = torch.linspace(0, self.max_step_size, self.n_step_size, device=device)
        # finding the locations on the phase mask
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu2(self.X, self.Y, self.SinX, self.SinY, self.L)

        # finding the gradient angle of the phase mask
        angle_x1, angle_y1 = find_mask_angles_gpu2_for_score(phase_mask_x, phase_mask_y, self.phase_maskx,
                                                             self.phase_masky, self.sampling_dist_mask_plane,
                                                             max_delta_x, max_delta_y, step_sizes, self.method)

        # finding the forward locations with the delta in the angle gradient
        # TODO: fix this line : CUDA out of memory
        mask_delta_x, mask_delta_y = find_forward_locations_gpu_parallel2_for_score(self.X, self.Y, self.SinX,
                                                                                    self.SinY, self.L,
                                                                                    angle_x1, angle_y1)
        mask_delta_x_shape = mask_delta_x[0].shape
        flattened_shape = (mask_delta_x_shape[0], mask_delta_x_shape[1] * mask_delta_x_shape[2],
                           mask_delta_x_shape[3] * mask_delta_x_shape[4])
        mask_delta_x = (mask_delta_x[0].reshape(flattened_shape), mask_delta_x[1].reshape(flattened_shape))
        mask_delta_y = (mask_delta_y[0].reshape(flattened_shape), mask_delta_y[1].reshape(flattened_shape))
        # interpolation of the mask
        mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
        mask_delta_y_loc = torch.stack(mask_delta_y, dim=-1)
        func = self.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(len(step_sizes), -1, -1, -1)
        weight_x = interpolator.grid_sample(func, mask_delta_x_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()
        weight_y = interpolator.grid_sample(func, mask_delta_y_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()

        # The cost
        weight_x = weight_x.reshape(mask_delta_x_shape)
        weight_y = weight_y.reshape(mask_delta_x_shape)
        weight_x = weight_x * lf
        weight_y = weight_y * lf

        # update the deltas' score

        score_x = weight_x
        score_y = weight_y

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices_x = torch.argmax(torch.sum(score_x, dim=(1, 2, 3, 4)))
        max_indices_y = torch.argmax(torch.sum(score_y, dim=(1, 2, 3, 4)))

        # Map the indices back to keys
        max_step_x = step_sizes[max_indices_x]
        max_step_y = step_sizes[max_indices_y]
        return max_step_x, max_step_y, torch.sum(score_x, dim=(1, 2, 3, 4))[max_indices_x], \
            torch.sum(score_y, dim=(1, 2, 3, 4))[max_indices_y]

    def _update_phase_mask_simple_gpu_parallel3(self, lf, max_delta_x, max_delta_y):
        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        step_sizes = torch.linspace(0, self.max_step_size, self.n_step_size, device=device)
        step_sizes = torch.cat((step_sizes, torch.tensor([1], device=device)))
        #step_sizes = torch.tensor([1, 0], device=device)
        size_to_score_dict = {}
        max_step_size = 0
        max_step_size_score = 0

        # finding the locations on the phase mask
        for step_size in step_sizes:
            phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu2(self.X, self.Y, self.SinX, self.SinY, self.L)

            phase_maskx = self.phase_maskx - step_size * max_delta_x
            phase_masky = self.phase_masky - step_size * max_delta_y

            # display(phase_maskx.cpu().numpy(), f"angle x with step {step_size}")
            # display(phase_masky.cpu().numpy(), f"angle y with step {step_size}")
            # plt.show()

            # finding the gradient angle of the phase mask
            angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                                       phase_maskx, phase_masky,
                                                       self.sampling_dist_mask_plane, self.method)

            # finding the forward locations with the delta in the angle gradient
            mask_delta_x, _ = find_forward_locations_gpu_parallel2(self.X, self.Y, self.SinX,
                                                                   self.SinY, self.L,
                                                                   angle_x1, angle_y1, torch.tensor([0], device=device))
            mask_delta_x_shape = mask_delta_x[0].shape
            flattened_shape = (mask_delta_x_shape[0], mask_delta_x_shape[1] * mask_delta_x_shape[2],
                               mask_delta_x_shape[3] * mask_delta_x_shape[4])
            mask_delta_x = (mask_delta_x[0].reshape(flattened_shape), mask_delta_x[1].reshape(flattened_shape))
            # interpolation of the mask
            mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
            func = self.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(1, -1, -1, -1)
            weight_x = interpolator.grid_sample(func, mask_delta_x_loc, mode=self.method,
                                                padding_mode='zeros',
                                                align_corners=True).squeeze()

            # The cost
            weight_x = weight_x.reshape(mask_delta_x_shape)
            # display_lf_summed(torch.squeeze(weight_x).cpu().numpy(), name=f'weight {step_size}')
            # plt.show()
            # display_lf_summed(weight_x.squeeze().cpu().numpy(), str(step_size))
            weight_x = weight_x * lf

            size_to_score_dict[float(step_size.cpu())] = torch.sum(weight_x)
            if torch.sum(weight_x) > max_step_size_score:
                max_step_size = step_size
                max_step_size_score = torch.sum(weight_x)

        print(size_to_score_dict)

        return max_step_size, max_step_size_score

    def single_iter_x_or_y_gpu_parallel2(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        deltas = torch.linspace(-self.max_delta, self.max_delta, self.n_delta, device=device)
        if 0 not in deltas:
            deltas = torch.concatenate((torch.tensor([0], device=device), deltas))
        else:
            zero_index = (deltas == 0).nonzero(as_tuple=True)[0].item()
            # Swap the zero to the beginning
            if zero_index != 0:  # Only rearrange if zero is not already at the beginning
                tensor = torch.cat((deltas[zero_index:zero_index + 1], deltas[:zero_index], deltas[zero_index + 1:]))
        # finding the locations on the phase mask
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu2(self.X, self.Y, self.SinX, self.SinY, self.L)

        # finding the gradient angle of the phase mask
        angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                                   self.phase_maskx, self.phase_masky,
                                                   self.sampling_dist_mask_plane, self.method)

        # finding the forward locations with the delta in the angle gradient
        mask_delta_x, mask_delta_y = find_forward_locations_gpu_parallel2(self.X, self.Y, self.SinX, self.SinY, self.L,
                                                                          angle_x1, angle_y1, deltas)
        mask_delta_x_shape = mask_delta_x[0].shape
        flattened_shape = (mask_delta_x_shape[0], mask_delta_x_shape[1] * mask_delta_x_shape[2],
                           mask_delta_x_shape[3] * mask_delta_x_shape[4])
        mask_delta_x = (mask_delta_x[0].reshape(flattened_shape), mask_delta_x[1].reshape(flattened_shape))
        mask_delta_y = (mask_delta_y[0].reshape(flattened_shape), mask_delta_y[1].reshape(flattened_shape))
        # interpolation of the mask
        mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
        mask_delta_y_loc = torch.stack(mask_delta_y, dim=-1)
        func = self.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(len(deltas), -1, -1, -1)
        weight_x = interpolator.grid_sample(func, mask_delta_x_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()
        weight_y = interpolator.grid_sample(func, mask_delta_y_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()

        # The cost
        weight_x = weight_x.reshape(mask_delta_x_shape)
        weight_y = weight_y.reshape(mask_delta_x_shape)


        weight_x = weight_x * lf
        weight_y = weight_y * lf


        weight_x = torch.transpose(torch.flatten(weight_x, start_dim=1), 0, 1)
        weight_y = torch.transpose(torch.flatten(weight_y, start_dim=1), 0, 1)

        # create transformation matrix to convert to the phase_mask shape
        transform_matrix = create_transform_matrix_gpu2(phase_mask_x, phase_mask_y,
                                                        self.sampling_dist_mask_plane,
                                                        self.phase_mask_shape, lf.shape)

        # update the deltas' score

        score_x = torch.transpose((transform_matrix @ weight_x), 0, 1).reshape(len(deltas),
                                                                               self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])

        score_y = torch.transpose((transform_matrix @ weight_y), 0, 1).reshape(len(deltas),
                                                                               self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])

        score_x = score_x.float()
        score_y = score_y.float()
        """
        for i in range(len(deltas)):
            score_x[i] = LPF(score_x[i], self.sigma)
            score_y[i] = LPF(score_y[i], self.sigma)"""

        # Find the indices of the maximum values along the new axis (axis=0)

        max_indices_x = torch.argmax(score_x, dim=0)
        max_indices_y = torch.argmax(score_y, dim=0)

        # Map the indices back to keys
        max_delta_x = torch.tensor(deltas)[max_indices_x]
        max_delta_y = torch.tensor(deltas)[max_indices_y]

        return self._update_phase_mask(lf, max_delta_x, max_delta_y)

    def single_iter_x_or_y_gpu_parallel2_debug(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        deltas = torch.linspace(-self.max_delta, self.max_delta, self.n_delta, device=device)
        if 0 not in deltas:
            deltas = torch.concatenate((torch.tensor([0], device=device), deltas))
        else:
            zero_index = (deltas == 0).nonzero(as_tuple=True)[0].item()
            # Swap the zero to the beginning
            if zero_index != 0:  # Only rearrange if zero is not already at the beginning
                tensor = torch.cat((deltas[zero_index:zero_index + 1], deltas[:zero_index], deltas[zero_index + 1:]))

        # finding the locations on the phase mask
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu2(self.X, self.Y, self.SinX, self.SinY, self.L)

        # finding the gradient angle of the phase mask
        angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                                   self.phase_maskx, self.phase_masky,
                                                   self.sampling_dist_mask_plane, self.method)

        # finding the forward locations with the delta in the angle gradient
        mask_delta_x, mask_delta_y = find_forward_locations_gpu_parallel2(self.X, self.Y, self.SinX, self.SinY, self.L,
                                                                          angle_x1, angle_y1, deltas)
        mask_delta_x_shape = mask_delta_x[0].shape
        flattened_shape = (mask_delta_x_shape[0], mask_delta_x_shape[1] * mask_delta_x_shape[2],
                           mask_delta_x_shape[3] * mask_delta_x_shape[4])
        mask_delta_x = (mask_delta_x[0].reshape(flattened_shape), mask_delta_x[1].reshape(flattened_shape))
        mask_delta_y = (mask_delta_y[0].reshape(flattened_shape), mask_delta_y[1].reshape(flattened_shape))
        # interpolation of the mask
        mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
        mask_delta_y_loc = torch.stack(mask_delta_y, dim=-1)
        func = self.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(len(deltas), -1, -1, -1)
        weight_x = interpolator.grid_sample(func, mask_delta_x_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()
        weight_y = interpolator.grid_sample(func, mask_delta_y_loc, mode=self.method,
                                            padding_mode='zeros',
                                            align_corners=True).squeeze()

        # The cost
        weight_x = weight_x.reshape(mask_delta_x_shape)
        weight_y = weight_y.reshape(mask_delta_x_shape)
        # display_lf_summed(weight_x[1].cpu().numpy(), "-1/6")
        # display_lf_summed(weight_x[-1].cpu().numpy(), "1/6")
        # plt.show()

        weight_x = weight_x * lf
        weight_y = weight_y * lf
        display_all_patches(phase_mask_x, phase_mask_y, self.sampling_dist_mask_plane, "test")
        alpha = display_lf_on_phase(phase_mask_x, phase_mask_y, self.sampling_dist_mask_plane, lf, "LF on Phase")

        weight_x = torch.transpose(torch.flatten(weight_x, start_dim=1), 0, 1)
        weight_y = torch.transpose(torch.flatten(weight_y, start_dim=1), 0, 1)

        # create transformation matrix to convert to the phase_mask shape
        transform_matrix = create_transform_matrix_gpu2(phase_mask_x, phase_mask_y,
                                                        self.sampling_dist_mask_plane,
                                                        self.phase_mask_shape, lf.shape)

        # update the deltas' score

        score_x = torch.transpose((transform_matrix @ weight_x), 0, 1).reshape(len(deltas),
                                                                               self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])

        # for debugging
        X = phase_mask_x[:, :, 1, 1].cpu().numpy()
        Y = phase_mask_y[:, :, 1, 1].cpu().numpy()
        X = (X + 1601 * self.sampling_dist_mask_plane / 2) / self.sampling_dist_mask_plane
        Y = (Y + 1601 * self.sampling_dist_mask_plane / 2) / self.sampling_dist_mask_plane
        bottom_left = (np.min(X), np.min(Y))
        top_right = (np.max(X), np.max(Y))
        display_with_opacity(score_x[0].cpu().numpy(), alpha, "0")
        plt.xlim(0, 1601)
        plt.ylim(1601, 0)
        # plt.xlim(bottom_left[0], top_right[0])  # Adjust as needed
        # plt.ylim(bottom_left[0], top_right[1])  # Adjust as needed

        display_with_opacity(score_x[1].cpu().numpy(), alpha, "1")
        plt.xlim(0, 1601)
        plt.ylim(1601, 0)
        # plt.xlim(bottom_left[0], top_right[0])  # Adjust as needed
        # plt.ylim(bottom_left[0], top_right[1])  # Adjust as needed
        display_with_opacity(score_x[2].cpu().numpy(), alpha, "2")
        plt.xlim(0, 1601)
        plt.ylim(1601, 0)
        # plt.xlim(bottom_left[0], top_right[0])  # Adjust as needed
        # plt.ylim(bottom_left[0], top_right[1])  # Adjust as needed

        display_with_opacity((score_x[2] > score_x[1]).cpu().numpy(), alpha, "2>1")
        plt.xlim(0, 1601)
        plt.ylim(1601, 0)
        display_with_opacity((score_x[0] > score_x[1]).cpu().numpy(), alpha, "0>1")
        plt.xlim(0, 1601)
        plt.ylim(1601, 0)
        # plt.xlim(bottom_left[0], top_right[0])  # Adjust as needed
        # plt.ylim(bottom_left[0], top_right[1])  # Adjust as needed
        score_y = torch.transpose((transform_matrix @ weight_y), 0, 1).reshape(len(deltas),
                                                                               self.phase_mask_shape[0],
                                                                               self.phase_mask_shape[1])

        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices_x = torch.argmax(score_x, dim=0)
        max_indices_y = torch.argmax(score_y, dim=0)

        # Map the indices back to keys
        max_delta_x = torch.tensor(deltas)[max_indices_x]
        max_delta_y = torch.tensor(deltas)[max_indices_y]
        # max_score = self._update_phase_mask(lf, max_delta_x, max_delta_y)
        # self._update_phase_mask_simple_gpu(lf, max_delta_x, max_delta_y)
        return self._update_phase_mask(lf, max_delta_x, max_delta_y)

    def single_iter_x_and_y_gpu_parallel2(self, lf):
        """
        A single iteration of the walker maximize the energy of the warped light field in the defined mask.
        The max delta is chosen, and added after iterating over all the LF angles.
        The scores are found for uncoupled delta_x and delta_y (independently)
        :param lf: the given light field
        :return: the energy in the mask
        """

        # create a list of possible x and y delta combinations
        device = torch.device("cuda")
        deltas_x = torch.linspace(-self.max_delta, self.max_delta, self.n_delta, device=device)

        if 0 not in deltas_x:
            deltas_x = torch.concatenate((torch.tensor([0], device=device), deltas_x))
        else:
            # TODO: Fix this
            zero_idx = torch.where(deltas_x==0)[0]
            deltas_x = torch.cat((torch.tensor([0], device=device), deltas_x[0:zero_idx],deltas_x[zero_idx+1:]))


        deltas_y = deltas_x
        # finding the locations on the phase mask
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu2(self.X, self.Y, self.SinX, self.SinY, self.L)

        # finding the gradient angle of the phase mask
        angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                                   self.phase_maskx, self.phase_masky,
                                                   self.sampling_dist_mask_plane, self.method)

        # finding the forward locations with the delta in the angle gradient
        mask_delta = find_forward_locations_gpu_parallel2_and(self.X, self.Y, self.SinX, self.SinY,
                                                              self.L,
                                                              angle_x1, angle_y1, deltas_x, deltas_y)
        mask_delta_shape = mask_delta[0].shape
        flattened_shape = (mask_delta_shape[0], mask_delta_shape[1] * mask_delta_shape[2],
                           mask_delta_shape[3] * mask_delta_shape[4])
        mask_delta = (mask_delta[0].reshape(flattened_shape), mask_delta[1].reshape(flattened_shape))
        # interpolation of the mask
        mask_delta_loc = torch.stack(mask_delta, dim=-1)
        func = self.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(len(deltas_x) ** 2, -1, -1, -1)
        weight = interpolator.grid_sample(func, mask_delta_loc, mode=self.method,
                                          padding_mode='zeros',
                                          align_corners=True).squeeze()

        # The cost
        weight = weight.reshape(mask_delta_shape)
        weight = weight * lf
        ## TODO: get rid of the transpose if possible
        weight = torch.transpose(torch.flatten(weight, start_dim=1), 0, 1)

        # create transformation matrix to convert to the phase_mask shape
        transform_matrix = create_transform_matrix_gpu2(phase_mask_x, phase_mask_y,
                                                        self.sampling_dist_mask_plane,
                                                        self.phase_mask_shape, lf.shape)

        # update the deltas' score

        score = torch.transpose((transform_matrix @ weight), 0, 1).reshape(len(deltas_x) ** 2,
                                                                           self.phase_mask_shape[0],
                                                                           self.phase_mask_shape[1])


        # Find the indices of the maximum values along the new axis (axis=0)
        max_indices = torch.argmax(score, dim=0)
        deltas = torch.cartesian_prod(deltas_x, deltas_y)

        # Map the indices back to keys
        max_delta = deltas[max_indices, :]
        max_delta_x = max_delta[:, :, 0]
        max_delta_y = max_delta[:, :, 1]

        # max_score = self._update_phase_mask(lf, max_delta_x, max_delta_y)
        return self._update_phase_mask(lf, max_delta_x, max_delta_y)


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
        # gradient_x = -torch.ones_like(gradient_x) * 1 / 6
        # gradient_y = -torch.ones_like(gradient_y) * 1 / 6
        gradient_x = LPF(gradient_x, self.sigma)
        gradient_y = LPF(gradient_y, self.sigma)
        max_step, score = self._update_phase_mask_simple_gpu_parallel3(lf, gradient_x, gradient_y)
        # max_step = 1
        # score = torch.tensor([0], device='cuda')
        print(f"found max step {max_step} with score of: {score}")
        # self.phase_maskx = -torch.ones_like(self.phase_maskx) * 1 / 6
        # self.phase_masky = -torch.ones_like(self.phase_masky) * 1 / 6
        self.phase_maskx = self.phase_maskx + max_step * gradient_x
        self.phase_masky = self.phase_masky + max_step * gradient_y

        return score

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
