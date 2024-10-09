import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from abc import abstractmethod
from angle_finder import angle_finder
from scipy.signal import convolve2d



class gradient_angle_finder(angle_finder):

    def __init__(self, sampling_dist, N, wavelength, sigma):
        self.sampling_dist = sampling_dist
        self.N = N
        self.wavelength = wavelength
        self.sigma = sigma
    def get_delta_sin(self, mask, filter=None):

        if filter is None:
            self.filter = self._create_filter()
        else:
            self.filter = filter
        self.filter = self.filter/np.sum(self.filter)
        display(self.filter, "Filter")


        delta_sin_x, delta_sin_y = self._get_sin_from_gradient(mask)
        display(np.unwrap(np.unwrap(np.angle(mask), axis=0), axis=1), "mask phase")
        display(delta_sin_x.T, "delta_sin_x")
        display(delta_sin_y.T, "delta_sin_y")
        return delta_sin_x.T, delta_sin_y.T

    def _create_filter(self):
        x_filter = np.linspace(0, self.N * self.sampling_dist, self.N) - self.N * self.sampling_dist / 2
        x_filter, y_filter = np.meshgrid(x_filter, x_filter)
        return np.exp(-(x_filter ** 2 + y_filter ** 2) / (2 * self.sigma ** 2))

    # when dividing into windows
    def _get_sin_from_gradient_old(self, mask):
        phase_map = np.unwrap(np.unwrap(np.angle(mask), axis=0), axis=1)  # fix 2pi jump
        angle_grad_x = np.gradient(phase_map, axis=1) / (
                self.sampling_dist * 2 * np.pi / self.wavelength)  # gives the actual sin_theta value
        angle_grad_y = np.gradient(phase_map, axis=0) / (
                self.sampling_dist * 2 * np.pi / self.wavelength)  # gives the actual sin_theta value
        # Padding for filter application (dividing into sections)

        N_full = phase_map.shape[0]
        pad_size = (self.N - (N_full % self.N)) % self.N  # Calculate the padding size
        # padding with 0's
        angle_grad_x = np.pad(angle_grad_x, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        angle_grad_y = np.pad(angle_grad_y, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
        N_blocks = angle_grad_y.shape[0] // self.N
        # Reshape to 4D array to separate the blocks
        angle_grad_x = angle_grad_x.reshape(N_blocks, self.N, N_blocks, self.N)
        angle_grad_y = angle_grad_y.reshape(N_blocks, self.N, N_blocks, self.N)

        # applying the specified filter
        filter = np.array(self.filter[np.newaxis, :, np.newaxis, :])
        filter = np.broadcast_to(filter, shape=angle_grad_x.shape)
        angle_grad_x = np.average(angle_grad_x, axis=(1, 3), weights=filter)
        angle_grad_y = np.average(angle_grad_y, axis=(1, 3), weights=filter)

        return angle_grad_x, angle_grad_y


    def _get_sin_from_gradient(self, mask):
        phase_map = np.unwrap(np.unwrap(np.angle(mask), axis=0), axis=1)  # fix 2pi jump
        angle_grad_x = np.gradient(phase_map, axis=1) / (
                self.sampling_dist * 2 * np.pi / self.wavelength)  # gives the actual sin_theta value
        angle_grad_y = np.gradient(phase_map, axis=0) / (
                self.sampling_dist * 2 * np.pi / self.wavelength)  # gives the actual sin_theta value
        # Padding for filter application (dividing into sections)
        """angle_grad_x = convolve2d(angle_grad_x, self.filter, mode="same")
        angle_grad_y = convolve2d(angle_grad_y, self.filter, mode="same")"""

        return angle_grad_x, angle_grad_y


