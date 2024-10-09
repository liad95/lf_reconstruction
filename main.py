import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *


def load_lf_data():
    # Load the .mat file
    lf = loadmat('lf.mat')['I_LF']
    mask = loadmat('mask.mat')['maskL']
    N_mask = int(np.sqrt(len(mask)))
    mask = np.reshape(mask, (N_mask, N_mask))
    # mask = np.transpose(mask)

    return lf, mask


def get_freq_change_from_mask(mask, sampling_dist, N, sigma, max_freq, n_angles):
    result = lgft(mask, sampling_dist, N, sigma, max_freq, n_angles)
    angle_map = get_angle_map_max(result)
    freq_map = convert_idx_to_freq(angle_map, max_freq, n_angles)
    return freq_map


def get_sin_change_from_mask(mask, sampling_dist, N, sigma, max_sin, n_angles, wavelength):
    max_freq = max_sin * 2 * np.pi / wavelength
    result = lgft(mask, sampling_dist, N, sigma, max_freq, n_angles)
    angle_map = get_angle_map_mean(result)
    sin_map = convert_idx_to_freq(angle_map, max_sin, n_angles)
    flattened_sin_map = sin_map.flatten()
    flattened_sin_map_x = np.array([element[0] for element in flattened_sin_map])
    flattened_sin_map_y = np.array([element[1] for element in flattened_sin_map])
    sin_map_x = flattened_sin_map_x.reshape(sin_map.shape)
    sin_map_y = flattened_sin_map_y.reshape(sin_map.shape)
    return sin_map_x, sin_map_y


def get_sin_from_gradient(mask, sampling_dist, N, wavelength, filter=None):
    phase_map = np.unwrap(np.unwrap(np.angle(mask), axis=0), axis=1)  # fix 2pi jump
    angle_grad_x = np.gradient(phase_map, axis=1) / (
            sampling_dist * 2 * np.pi / wavelength)  # gives the actual sin_theta value
    angle_grad_y = np.gradient(phase_map, axis=0) / (
            sampling_dist * 2 * np.pi / wavelength)  # gives the actual sin_theta value
    # Padding for filter application (dividing into sections)

    N_full = phase_map.shape[0]
    pad_size = (N - (N_full % N)) % N  # Calculate the padding size
    # padding with 0's
    angle_grad_x = np.pad(angle_grad_x, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
    angle_grad_y = np.pad(angle_grad_y, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
    N_blocks = angle_grad_y.shape[0] // N
    # Reshape to 4D array to separate the blocks
    angle_grad_x = angle_grad_x.reshape(N_blocks, N, N_blocks, N)
    angle_grad_y = angle_grad_y.reshape(N_blocks, N, N_blocks, N)

    if filter is None:
        # we just average if there is not filter specified
        angle_grad_x = np.average(angle_grad_x, axis=(1, 3))
        angle_grad_y = np.average(angle_grad_y, axis=(1, 3))
    else:
        # applying the specified filter
        filter = np.array(filter[np.newaxis, :, np.newaxis, :])
        filter = np.broadcast_to(filter, shape=angle_grad_x.shape)
        angle_grad_x = np.average(angle_grad_x, axis=(1, 3), weights=filter)
        angle_grad_y = np.average(angle_grad_y, axis=(1, 3), weights=filter)

    return angle_grad_x, angle_grad_y


def get_delta_sin(mask, sampling_dist, N, wavelength, filter=None):
    x_filter = np.linspace(0, N * sampling_dist_lf_plane, N) - N * sampling_dist_lf_plane / 2
    x_filter, y_filter = np.meshgrid(x_filter, x_filter)
    filter = np.exp(-(x_filter ** 2 + y_filter ** 2) / (2 * sigma ** 2))
    display(filter, "Filter")
    delta_sin_x, delta_sin_y = get_sin_from_gradient(mask, sampling_dist_mask_plane, N, wavelength, filter)

    display(np.unwrap(np.unwrap(np.angle(mask), axis=0), axis=1), "mask phase")
    display(delta_sin_x, "delta_sin_x")
    display(delta_sin_y, "delta_sin_y")
    return delta_sin_x, delta_sin_y


# region main


# params
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
sigma = 0.1
N = 10  # window size (N*N)
L = 100

# loading lf and mask, and extracting params
lf, mask = load_lf_data()
display_mask(mask, "Original")
display_lf_summed(lf, "Original")
lf_size = (lf.shape[0], lf.shape[1])
n_angles = lf.shape[2]

# This is the alternative to LGFT. Finding the phase gradient and applying a filter
delta_sin_x, delta_sin_y = get_delta_sin(mask, sampling_dist_mask_plane, N, wavelength)

# endregion

# region part 2
sampling_dist_delta_sin = sampling_dist_mask_plane * N  # This is if we work in the windows mode, where the number of elements now is (Nog/N)**2
delta_sin_size = delta_sin_x.shape
x_delta_sin = np.linspace(0, delta_sin_size[1] * sampling_dist_delta_sin, delta_sin_size[1]) - delta_sin_size[
    1] * sampling_dist_delta_sin / 2
y_delta_sin = np.linspace(0, delta_sin_size[0] * sampling_dist_delta_sin, delta_sin_size[0]) - delta_sin_size[
    0] * sampling_dist_delta_sin / 2

# creating the lf grid
sin_x = np.linspace(-max_sin, max_sin, n_angles)
sin_y = np.linspace(-max_sin, max_sin, n_angles)  # assuming sin_x = sin_y
x = np.linspace(0, lf_size[1] * sampling_dist_lf_plane, lf_size[1]) - lf_size[1] * sampling_dist_lf_plane / 2
y = np.linspace(0, lf_size[0] * sampling_dist_lf_plane, lf_size[0]) - lf_size[0] * sampling_dist_lf_plane / 2
X, Y, SinX, SinY = np.meshgrid(x, y, sin_x, sin_x, indexing='ij')
SinZ = np.sqrt(1 - np.power(SinX, 2) - np.power(SinY, 2))

inter1_points_x = X + L * SinX / SinZ
inter1_points_y = Y + L * SinY / SinZ

inter1_points = np.array([inter1_points_x.ravel(), inter1_points_y.ravel()]).T
delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), delta_sin_x, bounds_error=False)
delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), delta_sin_y, bounds_error=False)
delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(inter1_points_x.shape)
delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(inter1_points_x.shape)

# endregion

# region part 2

inter2_points_x = X - L * delta_sin_x_interp1
inter2_points_y = Y - L * delta_sin_y_interp1
inter2_points_sinx = SinX + delta_sin_x_interp1
inter2_points_siny = SinY + delta_sin_y_interp1
normalizing_factor = np.sqrt(np.power(inter2_points_sinx,2)+np.power(inter2_points_siny,2) + np.power(SinZ,2))
inter2_points_sinx = inter2_points_sinx/normalizing_factor
inter2_points_siny = inter2_points_siny/normalizing_factor

inter2_points = np.array(
    [inter2_points_x.ravel(), inter2_points_y.ravel(), inter2_points_sinx.ravel(), inter2_points_siny.ravel()]).T

lf_interp = RegularGridInterpolator((x, y, sin_x, sin_y), lf, bounds_error=False, fill_value=0)
lf_reconstructed = lf_interp(inter2_points).reshape(inter2_points_x.shape)

# endregion
display_lf_summed(lf_reconstructed, "Reconstructed")
display_lf_summed(lf, "Original")
display_lf_2d(lf, "Original")
display_lf_2d(lf_reconstructed, "Reconstructed")
