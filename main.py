import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import *


def load_lf_data():
    # Load the .mat file
    lf = loadmat('lf.mat')['I_LF']
    mask = loadmat('mask.mat')['maskL']
    N_mask = int(np.sqrt(len(mask)))
    mask = np.reshape(mask, (N_mask, N_mask))
    return lf, mask


"""def lf_shift(lf, angle_shift):
    # defining unknowns
    L = 1
    pixel_num_lf = lf.shape[0]
    pixel_num_mask = mask.shape[0]
    pixel_size_lf = 1
    pixel_size_mask = 1
    width_lf = pixel_size_lf * pixel_num_lf
    width_mask = pixel_size_mask * pixel_num_mask

    lf, mask = load_lf_data()
    phases = np.angle(mask)
    phase_x_grad, phase_y_grad = np.gradient(phases)
    pass"""


def no_mask_reconstruction(lf):
    max_angle = np.pi / 6
    angles = np.linspace(-max_angle, max_angle, 7)
    theta_x, theta_y = np.meshgrid(angles, angles)
    delta_x = -L


def generate_lgft_filter(sigma, wavelength, sample_dist, max_sin_angle, n_angles):
    k_x = np.linspace(-max_sin_angle, max_sin_angle, n_angles)
    k_x, k_y = np.meshgrid(k_x, k_x)
    k_x = (2 * np.pi / wavelength) * k_x
    k_y = (2 * np.pi / wavelength) * k_y
    N = 10
    x = (np.arange(0, N) - N / 2) * sample_dist
    x, y = np.meshgrid(x, x)
    gaussian = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * (sigma ** 2)))
    gaussian = gaussian / np.sum(gaussian)
    k_x = k_x[:, :, np.newaxis, np.newaxis]
    k_y = k_y[:, :, np.newaxis, np.newaxis]
    complex_exp = np.exp(1j * (k_x * x + k_y * y))
    filter = complex_exp * gaussian
    print((filter.shape))

    return filter


# local gaussian fourier tranform
# the distance between the sin_angle are linearly spaced. Not the distance between the angles
def lgft(matrix, sigma, wavelength, sample_dist, max_sin_angle, n_angles, N):
    filter = generate_dft_filter(sigma, wavelength, sample_dist, max_sin_angle, n_angles, N)
    filter = np.reshape(filter, (7, 7, -1))
    print(filter.shape)
    sample_matrix = np.reshape(matrix[0:N, 0:N], (N ** 2))
    print(sample_matrix.shape)
    result = np.dot(np.conj(filter), sample_matrix)
    print(result.shape)
    return result
    # N = (n_angles - 1) * np.floor(wavelength / sample_dist)


def display_lf(lf):
    image = np.sum(lf, axis=(2, 3))
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.show()


def display_mask(mask):
    plt.imshow(np.abs(mask), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.show()


N = 10
sample_dist = 2
X = generate_pure_freq(sample_dist, 0, 0, N)
display_mask(np.abs(np.fft.ifftshift((np.fft.fft2(X)))))
wavelength=4
max_freq = (2*np.pi/wavelength)*1/2
n_angles = 7
display_mask(np.abs(dft_specified_freq(X, 2, N, max_freq, n_angles)))
