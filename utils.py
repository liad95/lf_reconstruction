import numpy as np
import matplotlib.pyplot as plt

def generate_dft_filter(sample_dist, N):
    k = np.linspace(0, 2*np.pi/sample_dist, N, endpoint=False)
    k_x, k_y = np.meshgrid(k, k)
    x = (np.arange(0, N) - N / 2) * sample_dist
    x, y = np.meshgrid(x, x)
    k_x = k_x[:, :, np.newaxis, np.newaxis]
    k_y = k_y[:, :, np.newaxis, np.newaxis]
    complex_exp = np.exp(1j * (k_x * x + k_y * y))
    filter = complex_exp
    print((filter.shape))

    return filter

def generate_dft_filter_specified_freq(sample_dist, N, max_freq, n_freq):
    k = np.linspace(0, max_freq, n_freq, endpoint=False) - max_freq*(n_freq//2)/n_freq
    print(k)
    k_x, k_y = np.meshgrid(k, k)
    x = (np.arange(0, N) - N / 2) * sample_dist
    x, y = np.meshgrid(x, x)
    k_x = k_x[:, :, np.newaxis, np.newaxis]
    k_y = k_y[:, :, np.newaxis, np.newaxis]
    complex_exp = np.exp(1j * (k_x * x + k_y * y))
    filter = complex_exp
    print((filter.shape))

    return filter


def dft(matrix, sample_dist, N):
    filter = generate_dft_filter(sample_dist, N)
    filter = np.reshape(filter, (N,N,-1))
    print(filter.shape)
    print(matrix.shape)
    matrix = np.reshape(matrix, (N**2))
    print(matrix.shape)
    result = np.dot(np.conj(filter), matrix)
    print(result.shape)
    return result

def dft_specified_freq(matrix, sample_dist, N, max_freq, n_freq):
    filter = generate_dft_filter_specified_freq(sample_dist, N, max_freq, n_freq)
    filter = np.reshape(filter, (n_freq,n_freq,-1))
    print(filter.shape)
    print(matrix.shape)
    matrix = np.reshape(matrix, (N**2))
    print(matrix.shape)
    result = np.dot(np.conj(filter), matrix)
    print(result.shape)
    return result

def generate_pure_freq(sample_dist, k_x, k_y, N):
    x = np.linspace(0, N*sample_dist, N, endpoint=False)
    x, y = np.meshgrid(x, x)
    pure_freq = np.exp(1j * (k_x * x + k_y * y))

    plt.imshow(np.real(pure_freq), vmax=1, vmin=-1, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.show()
    return pure_freq
