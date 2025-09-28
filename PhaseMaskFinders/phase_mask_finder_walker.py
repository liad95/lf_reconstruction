import time

import matplotlib.pyplot as plt
import torch.nn.functional as F

from PhaseMaskFinders.phase_mask_finder import phase_mask_finder
from utils import *


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
            e.append(self.single_iter_phase_mask_finder(lf).cpu().numpy())
            print(f"iter #{k} - {e}")
            end_time = time.time()  # Record the end time
            iteration_time = end_time - start_time  # Calculate the time taken
            print(f"Iteration {k + 1} took {iteration_time:.4f} seconds")

        plt.figure()
        plt.plot(e)
        plt.show()
        return self.phase_maskx.cpu().numpy(), self.phase_masky.cpu().numpy()



    def _find_step_size_for_update(self, lf, max_delta_x, max_delta_y):
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
            phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X, self.Y, self.SinX, self.SinY, self.L)

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


    def single_iter_phase_mask_finder(self, lf):
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
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(self.X, self.Y, self.SinX, self.SinY, self.L)

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


        return self._update_phase_mask(lf, max_delta_x, max_delta_y)

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
        max_step, score = self._find_step_size_for_update(lf, gradient_x, gradient_y)
        # max_step = 1
        # score = torch.tensor([0], device='cuda')
        print(f"found max step {max_step} with score of: {score}")
        # self.phase_maskx = -torch.ones_like(self.phase_maskx) * 1 / 6
        # self.phase_masky = -torch.ones_like(self.phase_masky) * 1 / 6
        self.phase_maskx = self.phase_maskx + max_step * gradient_x
        self.phase_masky = self.phase_masky + max_step * gradient_y

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


