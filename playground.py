import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from skimage import data, color



def display_mask(mask):
    plt.imshow(np.abs(mask), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.show()





N = 16
sampling_dist = 0.25
wavelength = 0.5
max_sin = 0.5
max_freq = (2 * np.pi / wavelength) * max_sin
n_angles = 7
sigma = 10000
N_full = 1601
X_full = generate_pure_freq(sampling_dist, -2*1/12*2*np.pi, 0, N_full)
result = N*lgft(X_full, sampling_dist, N, sigma, max_freq, n_angles)
angle_map = get_angle_map_mean(result)
sin_map = convert_idx_to_freq(angle_map, max_sin, n_angles)


# converting to LF

inv_LF = convert_to_inv_LF(result, n_angles)
display_mask(inv_LF)
angle_map = get_angle_map_max(result)
print(angle_map)
freq_map = convert_idx_to_freq(angle_map, max_freq, n_angles)
print(freq_map)

LF = convert_to_LF(result, n_angles)
display_mask(LF)

angle_map_mean = get_angle_map_mean(result)
print(angle_map_mean)
print(convert_idx_to_freq(angle_map_mean, max_sin, n_angles))
