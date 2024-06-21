import numpy as np
import matplotlib.pyplot as plt


def generate_pure_freq(sample_dist, k_x, k_y, N):
    x = np.linspace(0, N * sample_dist, N, endpoint=False)
    x, y = np.meshgrid(x, x)
    pure_freq = np.exp(1j * (k_x * x + k_y * y))

    plt.imshow(np.real(pure_freq), vmax=1, vmin=-1, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.show()
    return pure_freq

# region LGFT

def gft(matrix, sample_dist, N, sigma, max_freq=None, n_freq=None):
    if n_freq is None:
        n_freq = N

    filter = generate_gft_filter(sample_dist, N, N, sigma, max_freq, n_freq)
    filter = np.reshape(filter, (n_freq, n_freq, -1))
    matrix = np.reshape(matrix, (N ** 2))
    result = np.dot(np.conj(filter), matrix)
    return result


def generate_gft_filter(sample_dist, Nx, Ny, sigma, max_freq, n_freq):
    if max_freq is None:
        max_freq = np.pi / sample_dist
        k = np.linspace(0, 2 * max_freq, n_freq, endpoint=False)
    else:
        k = np.linspace(0, 2 * max_freq, n_freq, endpoint=True)
    delta_k = k[1]
    k = k - max_freq
    k_x, k_y = np.meshgrid(k, k)
    x = (np.arange(0, Nx) - Nx / 2) * sample_dist
    y = (np.arange(0, Ny) - Ny / 2) * sample_dist
    x, y = np.meshgrid(x, y)
    k_x = k_x[:, :, np.newaxis, np.newaxis]
    k_y = k_y[:, :, np.newaxis, np.newaxis]
    complex_exp = np.exp(-1j * (k_x * x + k_y * y))
    gaussian = np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * (sigma ** 2)))
    gaussian = gaussian / np.sum(gaussian)
    """print(complex_exp.shape)
    plt.imshow(np.real(complex_exp[2,2,:, :]))
    plt.show()"""
    filter = complex_exp * gaussian

    return filter


def dft(matrix, sample_dist, N, max_freq=None, n_freq=None):
    return gft(matrix, sample_dist, N, np.inf, max_freq, n_freq)


def lgft(matrix, sample_dist, N, sigma, max_freq=None, n_freq=None):
    height = matrix.shape[0]
    width = matrix.shape[1]
    Ny_window = height // N
    Nx_window = width // N
    result = np.zeros((n_freq, n_freq, Ny_window, Nx_window)).astype(complex)
    for i in range(Nx_window):
        for j in range(Ny_window):
            local_window = matrix[i * N:(i + 1) * N, j * N:(j + 1) * N]
            result[:, :, i, j] = gft(local_window, sample_dist, N, sigma, max_freq, n_freq)

    return result


# endregion

# region Convert to LF
def convert_to_inv_LF(matrix, n_angles):
    new_shape = (n_angles * matrix.shape[2], n_angles * matrix.shape[3])
    inv_LF = matrix.transpose(0, 2, 1, 3).reshape(new_shape)
    return inv_LF


def convert_to_LF(matrix, n_angles):
    new_shape = (n_angles * matrix.shape[2], n_angles * matrix.shape[3])
    LF = matrix.transpose(2, 0, 3, 1).reshape(new_shape)
    return LF

# endregion

#region get frequency map
def get_angle_map_max(lgft_result):
    max_angles = np.empty((lgft_result.shape[2], lgft_result.shape[3]), dtype=tuple)
    for i in range(max_angles.shape[0]):
        for j in range(max_angles.shape[1]):
            gft_result = np.abs(lgft_result[:, :, i, j])
            max_angles[i, j] = np.unravel_index(np.argmax(gft_result), gft_result.shape, order='F')
    return max_angles


def get_angle_map_mean(lgft_result):
    mean_angles = np.empty((lgft_result.shape[2], lgft_result.shape[3]), dtype=tuple)
    indices_x = np.arange(lgft_result.shape[0])[np.newaxis, :].repeat(lgft_result.shape[1], axis=0)
    indices_y = np.arange(lgft_result.shape[1])[:, np.newaxis].repeat(lgft_result.shape[0], axis=1)
    for i in range(mean_angles.shape[0]):
        for j in range(mean_angles.shape[1]):
            gft_result = np.abs(lgft_result[:, :, i, j])
            normalization_factor = np.sum(gft_result)
            angle_x = np.sum(indices_x * gft_result) / normalization_factor
            angle_y = np.sum(indices_y * gft_result) / normalization_factor

            mean_angles[i, j] = (angle_x, angle_y)
    return mean_angles


def convert_idx_to_freq(angle_map, max_freq, n_angles):
    freq = np.empty_like(angle_map)
    shift = n_angles - 1 - n_angles // 2
    coef = max_freq / shift

    for i in range(angle_map.shape[0]):
        for j in range(angle_map.shape[1]):
            freq_x = coef * (angle_map[i, j][0] - shift)
            freq_y = coef * (angle_map[i, j][1] - shift)
            freq[i, j] = (freq_x, freq_y)
    return freq

#endregion