import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import linalg
from cupyx.scipy.sparse import csr_matrix as csr_gpu
from cupyx.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorGPU
import cupy as cp
from scipy.sparse import identity


def generate_pure_freq(sample_dist, k_x, k_y, N):
    x = np.linspace(0, N * sample_dist, N, endpoint=False)
    x, y = np.meshgrid(x, x)
    pure_freq = np.exp(1j * (k_x * x + k_y * y))

    plt.imshow(np.real(pure_freq), vmax=1, vmin=-1, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar

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

# region get frequency map
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


# endregion
def find_phase_mask_locations_gpu(X, Y, SinX, SinY, L):
    SinZ = cp.sqrt(1 - cp.power(SinX, 2) - cp.power(SinY, 2))
    inter1_points_x = X + L * SinX / SinZ
    inter1_points_y = Y + L * SinY / SinZ
    return inter1_points_x, inter1_points_y


def find_phase_mask_locations(X, Y, SinX, SinY, L):
    SinZ = np.sqrt(1 - np.power(SinX, 2) - np.power(SinY, 2))
    inter1_points_x = X + L * SinX / SinZ
    inter1_points_y = Y + L * SinY / SinZ
    return inter1_points_x, inter1_points_y


def find_mask_angles_gpu(inter1_points_x, inter1_points_y, maskx, masky, sampling_dist_mask_plane):
    delta_sin_size = maskx.shape
    x_delta_sin = cp.linspace(0, delta_sin_size[1] * sampling_dist_mask_plane, delta_sin_size[1],
                              endpoint=False) - \
                  (delta_sin_size[1] - 1) * sampling_dist_mask_plane / 2
    y_delta_sin = cp.linspace(0, delta_sin_size[0] * sampling_dist_mask_plane, delta_sin_size[0],
                              endpoint=False) - \
                  (delta_sin_size[0] - 1) * sampling_dist_mask_plane / 2

    inter1_points = cp.array([inter1_points_x.ravel(), inter1_points_y.ravel()]).T
    delta_sin_x_func = RegularGridInterpolatorGPU((x_delta_sin, y_delta_sin), maskx, bounds_error=False)
    delta_sin_y_func = RegularGridInterpolatorGPU((x_delta_sin, y_delta_sin), masky, bounds_error=False)
    delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(inter1_points_x.shape)
    delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(inter1_points_x.shape)
    return delta_sin_x_interp1, delta_sin_y_interp1


def find_mask_angles(inter1_points_x, inter1_points_y, maskx, masky, sampling_dist_mask_plane, method='linear'):
    delta_sin_size = maskx.shape
    x_delta_sin = np.linspace(0, delta_sin_size[1] * sampling_dist_mask_plane, delta_sin_size[1],
                              endpoint=False) - \
                  (delta_sin_size[1] - 1) * sampling_dist_mask_plane / 2
    y_delta_sin = np.linspace(0, delta_sin_size[0] * sampling_dist_mask_plane, delta_sin_size[0],
                              endpoint=False) - \
                  (delta_sin_size[0] - 1) * sampling_dist_mask_plane / 2

    inter1_points = np.array([inter1_points_x.ravel(), inter1_points_y.ravel()]).T
    delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), maskx, bounds_error=False, method=method)
    delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), masky, bounds_error=False, method=method)
    delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(inter1_points_x.shape)
    delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(inter1_points_x.shape)
    return delta_sin_x_interp1, delta_sin_y_interp1


def find_forward_locations(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1):
    # endregion
    # region part 2
    SinZ = np.sqrt(1 - np.power(SinX, 2) - np.power(SinY, 2))
    inter2_points_sinx = SinX - delta_sin_x_interp1
    inter2_points_siny = SinY - delta_sin_y_interp1
    sinz_squared = 1 - np.power(inter2_points_sinx, 2) - np.power(inter2_points_siny, 2)
    sinz_squared = np.where(sinz_squared <= 0, 0.01, sinz_squared)
    sinz = np.sqrt(sinz_squared)
    inter2_points_x = X - L * (inter2_points_sinx / sinz - SinX / SinZ)
    inter2_points_y = Y - L * (inter2_points_siny / sinz - SinY / SinZ)
    return inter2_points_x, inter2_points_y


def find_forward_locations_gpu(X, Y, SinX, SinY, L, delta_sin_x_interp1, delta_sin_y_interp1):
    # endregion
    # region part 2
    SinZ = cp.sqrt(1 - cp.power(SinX, 2) - cp.power(SinY, 2))
    inter2_points_sinx = SinX - delta_sin_x_interp1
    inter2_points_siny = SinY - delta_sin_y_interp1
    sinz = cp.sqrt(1 - cp.power(inter2_points_sinx, 2) - cp.power(inter2_points_siny, 2))
    inter2_points_x = X - L * (inter2_points_sinx / sinz - SinX / SinZ)
    inter2_points_y = Y - L * (inter2_points_siny / sinz - SinY / SinZ)
    return inter2_points_x, inter2_points_y


def point_in_grid(x, y, height, width):
    # is_in_grid = np.zeros_like(x, dtype=bool)
    is_in_grid = np.logical_and.reduce([x >= 0, x < width, y >= 0, y < height])
    return is_in_grid


def point_in_grid_gpu(x, y, height, width):
    # is_in_grid = np.zeros_like(x, dtype=bool)
    is_in_grid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return is_in_grid


def create_weighted_forward(interp_x, interp_y, sampling_dist, matrix):
    height = matrix.shape[0]
    width = matrix.shape[1]
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height * sampling_dist / 2) / sampling_dist
    high_x = np.ceil(inter2_points_x)
    low_x = np.floor(inter2_points_x)
    high_y = np.ceil(inter2_points_y)
    low_y = np.floor(inter2_points_y)

    is_valid = np.concatenate((point_in_grid(low_x, low_y, height, width), point_in_grid(high_x, low_y, height, width),
                               point_in_grid(low_x, high_y, height, width),
                               point_in_grid(high_x, high_y, height, width)))
    upper_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    weights = np.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    upper_right_idx = high_x + high_y * width
    upper_left_idx = low_x + high_y * width
    lower_right_idx = high_x + low_y * width
    lower_left_idx = low_x + low_y * width
    row_idx = np.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = np.squeeze(np.argwhere(is_valid))
    col_idx = np.tile(np.arange(height * width), 4)
    weight_matrix = scipy.sparse.csr_array((weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                                           shape=(height * width, height * width))
    # row_sums = np.array(weight_matrix.sum(axis=1))  # Sum of each row
    # row_indices, col_indices = weight_matrix.nonzero()o
    # weight_matrix.data /= row_sums[row_indices]

    matrix = matrix.flatten()
    # weight_matrix = identity(height * width, format='csr')
    ## TODO: fix the transpose of the next line
    weighted_matrix = (weight_matrix @ matrix).reshape(height, width)
    return weighted_matrix


def create_weighted_forward_for_phase_mask(interp_x, interp_y, sampling_dist, matrix, phase_mask_shape):
    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix.shape[0]
    width_matrix = matrix.shape[1]
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist
    high_x = np.ceil(inter2_points_x)
    low_x = np.floor(inter2_points_x)
    high_y = np.ceil(inter2_points_y)
    low_y = np.floor(inter2_points_y)

    is_valid = np.concatenate((point_in_grid(low_x, low_y, height_phase, width_phase),
                               point_in_grid(high_x, low_y, height_phase, width_phase),
                               point_in_grid(low_x, high_y, height_phase, width_phase),
                               point_in_grid(high_x, high_y, height_phase, width_phase)))
    upper_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    weights = np.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = np.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = np.squeeze(np.argwhere(is_valid))
    col_idx = np.tile(np.arange(height_matrix * width_matrix), 4)
    weight_matrix = scipy.sparse.csr_array((weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                                           shape=(height_phase * width_phase, height_matrix * width_matrix))

    weight_matrix_gpu = csr_gpu(weight_matrix)
    matrix_gpu = cp.array(matrix.flatten())
    weighted_matrix = (weight_matrix_gpu.dot(matrix_gpu)).reshape(height_phase, width_phase)
    return cp.asnumpy(weighted_matrix)


def create_transform_matrix(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist
    high_x = np.ceil(inter2_points_x)
    low_x = np.floor(inter2_points_x)
    high_y = np.ceil(inter2_points_y)
    low_y = np.floor(inter2_points_y)

    is_valid = np.concatenate((point_in_grid(low_x, low_y, height_phase, width_phase),
                               point_in_grid(high_x, low_y, height_phase, width_phase),
                               point_in_grid(low_x, high_y, height_phase, width_phase),
                               point_in_grid(high_x, high_y, height_phase, width_phase)))
    upper_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - np.abs(high_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - np.abs(low_x - inter2_points_x)) * (1 - np.abs(low_y - inter2_points_y))
    weights = np.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = np.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = np.squeeze(np.argwhere(is_valid))
    col_idx = np.tile(np.arange(height_matrix * width_matrix), 4)
    weight_matrix = scipy.sparse.csr_array((weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                                           shape=(height_phase * width_phase, height_matrix * width_matrix))

    return weight_matrix


def create_transform_matrix_gpu(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist
    high_x = cp.ceil(inter2_points_x)
    low_x = cp.floor(inter2_points_x)
    high_y = cp.ceil(inter2_points_y)
    low_y = cp.floor(inter2_points_y)

    is_valid = cp.concatenate((point_in_grid_gpu(low_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(low_x, high_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, high_y, height_phase, width_phase)))
    upper_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    weights = cp.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = cp.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = cp.squeeze(cp.argwhere(is_valid))
    col_idx = cp.tile(cp.arange(height_matrix * width_matrix), 4)
    weight_matrix = csr_gpu((weights[valid_idx], (row_idx[valid_idx], col_idx[valid_idx])),
                            shape=(height_phase * width_phase, height_matrix * width_matrix))

    return weight_matrix


def create_transform_matrix_gpu2(interp_x, interp_y, sampling_dist, phase_mask_shape, matrix_shape):
    height_phase = phase_mask_shape[0]
    width_phase = phase_mask_shape[1]
    height_matrix = matrix_shape[0]
    width_matrix = matrix_shape[1]
    inter2_points_x = interp_x.flatten()
    inter2_points_x = (inter2_points_x + width_phase * sampling_dist / 2) / sampling_dist
    inter2_points_y = interp_y.flatten()
    inter2_points_y = (inter2_points_y + height_phase * sampling_dist / 2) / sampling_dist
    high_x = cp.ceil(inter2_points_x)
    low_x = cp.floor(inter2_points_x)
    high_y = cp.ceil(inter2_points_y)
    low_y = cp.floor(inter2_points_y)

    is_valid = cp.concatenate((point_in_grid_gpu(low_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, low_y, height_phase, width_phase),
                               point_in_grid_gpu(low_x, high_y, height_phase, width_phase),
                               point_in_grid_gpu(high_x, high_y, height_phase, width_phase)))
    upper_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    upper_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(high_y - inter2_points_y))
    lower_right_weight = (1 - cp.abs(high_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    lower_left_weight = (1 - cp.abs(low_x - inter2_points_x)) * (1 - cp.abs(low_y - inter2_points_y))
    weights = cp.concatenate((lower_left_weight, lower_right_weight, upper_left_weight, upper_right_weight))

    upper_right_idx = high_x + high_y * width_phase
    upper_left_idx = low_x + high_y * width_phase
    lower_right_idx = high_x + low_y * width_phase
    lower_left_idx = low_x + low_y * width_phase
    row_idx = cp.concatenate(
        (lower_left_idx, lower_right_idx, upper_left_idx, upper_right_idx)).astype(
        int)
    valid_idx = cp.squeeze(cp.argwhere(is_valid))
    col_idx = cp.tile(cp.arange(height_matrix * width_matrix), 4)

    return weights[valid_idx], row_idx[valid_idx], col_idx[valid_idx]
