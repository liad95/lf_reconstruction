import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
from AngleFinders.gradient_angle_finder import gradient_angle_finder
from Reconstructors.lf_forward_reconstruction import lf_forward_reconstructor
import torch
import gc
import numpy as np
from display import *
from utils import *
from scipy.interpolate import interp2d


# region memory debug utils
def get_tensor_memory(tensor):
    if tensor.is_cuda:
        if isinstance(tensor, torch.sparse.Tensor) and tensor.is_sparse and tensor.layout == torch.sparse_coo:
            return get_sparse_tensor_memory(tensor)
        else:
            size_in_bytes = tensor.element_size() * tensor.nelement()
            size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
            return size_in_mb
    return 0


def get_sparse_tensor_memory(sparse_matrix):
    # Get the number of non-zero elements
    num_non_zero = sparse_matrix._nnz()

    # Get the data type of the elements
    data_type = sparse_matrix.dtype

    # Calculate the size in bytes based on the data type
    size_per_element = torch.tensor(0, dtype=data_type).element_size()

    # Total size in bytes
    total_size_bytes = num_non_zero * size_per_element

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 ** 2)

    return total_size_mb


def get_gpu_memory_status():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()

        # Get total, allocated, and reserved memory
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        memory_allocated = torch.cuda.memory_allocated(gpu_id)
        memory_reserved = torch.cuda.memory_reserved(gpu_id)

        # Format the output to return memory in GB

        print(f"total_memory: {total_memory / (1024 ** 3)}")
        print(f"memory_allocated: {memory_allocated / (1024 ** 3)}")
        print(f"memory_reserved: {memory_reserved / (1024 ** 3)}")
    else:
        raise RuntimeError("CUDA is not available on this system.")


class tensor_dict:
    def __init__(self):
        self.dict = {}

    def track_obj(self, obj, name):
        self.dict[id(obj)] = (name, get_tensor_memory(obj))

    # Define a function to check tensor memory

    def print_active_tensors(self):
        total_memory = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
                if id(obj) in self.dict:
                    value = self.dict[id(obj)]
                    print(f"Tensor {(value[0])}: {value[1]:.2f} MB")
                    total_memory += value[1]
        print(f"Total memory: {total_memory} MB \n")

    def update_active_tensors(self):
        total_memory = 0
        exists = dict.fromkeys(self.dict.keys(), 0)
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
                if id(obj) in self.dict:
                    tensor_memory = get_tensor_memory(obj)
                    self.dict[id(obj)] = (self.dict[id(obj)][0], tensor_memory)
                    total_memory += tensor_memory
                    exists[id(obj)] = 1

        for key, exist in exists.items():
            if not exist:
                del self.dict[key]


# endregion

def find_phase_mask_loc_relevant_to_recon_point(x, y, kx, ky, max_sin, lf, sampling_dist_lf, L, phase_mask_x,
                                                phase_mask_y):
    """
    Displays the intersection point of a LF pixel with the phase mask
    :param x: the x pixel in the LF
    :param y: the y pixel in the LF
    :param kx: the kx pixel in the LF
    :param ky: the ky pixel in the LF
    :param max_sin: the max sine of the LF
    :param lf: the original LF
    :param sampling_dist_lf: the sampling distance of the LF
    :param L: the distance between the LF and the phase mask
    :param phase_mask_x: the phase mask in x
    :param phase_mask_y: the phase mask in y
    """
    lf_shape = lf.shape
    vKx = np.linspace(-max_sin + 1 / 14, max_sin - 1 / 14, lf_shape[3])
    vKy = np.linspace(-max_sin + 1 / 14, max_sin - 1 / 14, lf_shape[2])
    vX = np.linspace(0, lf_shape[1] * sampling_dist_lf,
                     lf_shape[1], endpoint=False) - (lf_shape[1] - 1) * sampling_dist_lf / 2
    vY = np.linspace(0, lf_shape[0] * sampling_dist_lf,
                     lf_shape[0], endpoint=False) - (lf_shape[0] - 1) * sampling_dist_lf / 2
    x = vX[x]
    y = vY[y]
    kx = vKx[kx]
    ky = vKy[ky]

    SinZ = np.sqrt(1 - (kx ** 2) - (ky ** 2))
    inter1_points_x = x + L * kx / SinZ
    inter1_points_y = y + L * ky / SinZ

    inter1_points_x = inter1_points_x + 800
    inter1_points_y = inter1_points_y + 800
    display(phase_mask_x, "Phase Mask X")
    plt.scatter(inter1_points_x, inter1_points_y, 1, 'r')
    display(phase_mask_y, "Phase Mask Y")
    plt.scatter(inter1_points_x, inter1_points_y, 1, 'r')


def forward_warp_a_pixel(x, y, kx, ky, max_sin, lf, sampling_dist_lf, L, phase_mask_x, phase_mask_y,
                         sampling_dist_mask_plane):
    lf_shape = lf.shape
    vKx = np.linspace(-max_sin + 1 / 14, max_sin - 1 / 14, lf_shape[3])
    vKy = np.linspace(-max_sin + 1 / 14, max_sin - 1 / 14, lf_shape[2])
    vX = np.linspace(0, lf_shape[1] * sampling_dist_lf,
                     lf_shape[1], endpoint=False) - (lf_shape[1] - 1) * sampling_dist_lf / 2
    vY = np.linspace(0, lf_shape[0] * sampling_dist_lf,
                     lf_shape[0], endpoint=False) - (lf_shape[0] - 1) * sampling_dist_lf / 2

    phase_mask_shape = phase_mask_x.shape
    vPhasex = np.linspace(0, phase_mask_shape[1] * sampling_dist_mask_plane,
                          phase_mask_shape[1], endpoint=False) - (
                      phase_mask_shape[1] - 1) * sampling_dist_mask_plane / 2
    vPhasey = np.linspace(0, phase_mask_shape[0] * sampling_dist_mask_plane,
                          phase_mask_shape[0], endpoint=False) - (
                      phase_mask_shape[0] - 1) * sampling_dist_mask_plane / 2
    x = vX[x]
    y = vY[y]
    kx = vKx[kx]
    ky = vKy[ky]

    SinZ = np.sqrt(1 - (kx ** 2) - (ky ** 2))
    inter1_points_x = x + L * kx / SinZ
    inter1_points_y = y + L * ky / SinZ

    f = interp2d(vPhasex, vPhasey, phase_mask_x)
    delta_k_x = f(inter1_points_x, inter1_points_y)
    f = interp2d(vPhasex, vPhasey, phase_mask_y)
    delta_k_y = f(inter1_points_x, inter1_points_y)

    kx_new = kx + delta_k_x
    ky_new = ky + delta_k_y
    sinz_new = np.sqrt(1 - kx_new ** 2 - ky_new ** 2)
    x = x + L * (kx / SinZ - kx_new / sinz_new)
    y = y + L * (ky / SinZ - ky_new / sinz_new)
    x = x / sampling_dist_lf + (lf_shape[0] - 1) / 2
    y = y / sampling_dist_lf + (lf_shape[1] - 1) / 2
    return x, y


def display_forwarded_loc(x, y, max_sin, lf, sampling_dist_lf, L, phase_mask_x, phase_mask_y, sampling_dist_mask_plane):
    """
    Displays the location of the forwarded pixel in the LF. Displays for each angle in the LF. Displays a rect of the bounds of the reconstructed LF
    The alpha of the scattered pixels is dependent on the LF pixel's energy
    :param x: the x pixel in the LF
    :param y: the y pixel in the LF
    :param max_sin: the max sine of the LF
    :param lf: the original LF
    :param sampling_dist_lf: the sampling distance of the LF
    :param L: the distance between the LF and the phase mask
    :param phase_mask_x: the phase mask in x
    :param phase_mask_y: the phase mask in y
    :param sampling_dist_mask_plane: the sampling distance of the phase mask
    """
    lf_shape = lf.shape
    plt.figure()
    alpha_factor = np.max(lf)
    plt.title("Forwarded LF Location")
    for i in range(lf_shape[2]):
        for j in range(lf_shape[2]):
            x_forwarded, y_forwarded = forward_warp_a_pixel(x, y, i, j, max_sin, lf, sampling_dist_lf, L, phase_mask_x,
                                                            phase_mask_y, sampling_dist_mask_plane)
            plt.scatter(x_forwarded, y_forwarded, alpha=lf[x, y, i, j] / alpha_factor)
    rect = patches.Rectangle((0, 0), lf_shape[0], lf_shape[0], linewidth=2, edgecolor='r', facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.legend()
    plt.show()

def display_FW_for_different_step_sizes(finder, step_sizes, lf, max_delta_x, max_delta_y, N, isScaled = False, isWeighted = False):
    """
    Displays the forwarded LF for different step sizes.  Also, displays a graph of the score Vs step size
    :param finder: the phase_mask_finder_walker obj
    :param step_sizes: the step sizes to test
    :param lf: the original LF, for score calculation
    :param max_delta_x: the found max delta in the x phase
    :param max_delta_y: the found max delta in the y phase
    :param N:
    :param isScale: whether to scale all the image colors together
    :param isWeighted: whether to display the forward LF weighted using the Mask
    """
    phasex = finder.phase_maskx.cpu().numpy()
    phasey = finder.phase_masky.cpu().numpy()
    gradientx = max_delta_x.cpu().numpy()
    gradienty = max_delta_y.cpu().numpy()
    lf = lf.cpu().numpy()
    step_sizes = step_sizes.cpu().numpy()
    mask = finder.mask.cpu().numpy()

    angle_finder = gradient_angle_finder(
        finder.sampling_dist_mask_plane, N, finder.wavelength, finder.sigma
    )
    reconstructor = lf_forward_reconstructor(
        finder.max_sin, finder.wavelength, finder.sampling_dist_lf_plane,
        finder.sampling_dist_mask_plane, N, finder.L, angle_finder, lf.shape
    )

    # Figure layout
    num_plots = len(step_sizes)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Shared normalization
    norm = mcolors.Normalize()
    ims = []
    scores = np.array([])
    global_min, global_max = np.inf, -np.inf

    # ---- Main loop ----
    for idx, step_size in enumerate(step_sizes):
        gradientx_new = phasex + step_size * gradientx
        gradienty_new = phasey + step_size * gradienty
        lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(
            lf, -gradientx_new, -gradienty_new
        )
        score = np.sum(lf_reconstructed_gradient * mask)

        if isWeighted:
            image = lf_reconstructed_gradient * mask
        else:
            image = lf_reconstructed_gradient
        if image.ndim >= 4:
            image = np.sum(image, axis=(2, 3))

        # track global min/max
        global_min = min(global_min, image.min())
        global_max = max(global_max, image.max())

        # plot with shared norm
        ax = axes_flat[idx]
        if isScaled:
            im = ax.imshow(image, cmap='viridis', interpolation='nearest', norm=norm)
        else:
            im = ax.imshow(image, cmap='viridis', interpolation='nearest')
        ims.append(im)
        ax.set_title(f"step = {float(step_size):.3g}\nscore = {float(score):.3g}")
        ax.axis('off')

        scores = np.append(scores, score)

    # hide unused subplots
    for k in range(len(scores), len(axes_flat)):
        axes_flat[k].axis('off')

    # apply global scaling to all images
    norm.vmin, norm.vmax = global_min, global_max

    # single colorbar
    fig.colorbar(ims[0], ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.04)

    fig.suptitle("Weights per step size", fontsize=12)
    plt.tight_layout()

    # plot scores
    plt.figure()
    sorted_indices = np.argsort(step_sizes)
    plt.plot(step_sizes[sorted_indices], scores[sorted_indices])
    plt.title("Scores Vs Step Size")


def display_step_sizes_find_process(finder, max_delta_x, max_delta_y, lf, step_sizes, isScale=False, isWeighted = False):
    """
    Displays the forwarded mask, for the different step sizes. Also, displays a graph of the score Vs step size
    :param finder: the phase_mask_finder_walker obj.
    :param max_delta_x: the found max delta in the x phase
    :param max_delta_y: the found max delta in the y phase
    :param lf: the original LF, for score calculation
    :param step_sizes: the step sizes to test
    :param isScale: whether to scale all the image colors together
    :param isWeighted: whether to display the forward mask weighted using the original LF
    """
    device = torch.device("cuda")
    scores = np.array([])

    # Prepare figure layout
    num_plots = len(step_sizes)
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Shared normalization for all images
    norm = mcolors.Normalize()
    ims = []
    global_min, global_max = np.inf, -np.inf

    # ---- Main loop: compute + plot ----
    for idx, step_size in enumerate(step_sizes):
        phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(
            finder.X, finder.Y, finder.SinX, finder.SinY, finder.L
        )

        phase_maskx = finder.phase_maskx + step_size * max_delta_x
        phase_masky = finder.phase_masky + step_size * max_delta_y

        angle_x1, angle_y1 = find_mask_angles_gpu2(
            phase_mask_x, phase_mask_y,
            phase_maskx, phase_masky,
            finder.sampling_dist_mask_plane, finder.method
        )

        mask_delta_x, _ = find_forward_locations_gpu_parallel2(
            finder.X, finder.Y, finder.SinX, finder.SinY, finder.L,
            angle_x1, angle_y1, torch.tensor([0], device=device)
        )
        mask_delta_x_shape = mask_delta_x[0].shape
        flattened_shape = (
            mask_delta_x_shape[0],
            mask_delta_x_shape[1] * mask_delta_x_shape[2],
            mask_delta_x_shape[3] * mask_delta_x_shape[4]
        )
        mask_delta_x = (
            mask_delta_x[0].reshape(flattened_shape),
            mask_delta_x[1].reshape(flattened_shape)
        )

        mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
        func = finder.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(1, -1, -1, -1)
        weight_x = interpolator.grid_sample(
            func, mask_delta_x_loc, mode=finder.method,
            padding_mode='zeros', align_corners=True
        ).squeeze()

        weight_x = weight_x.reshape(mask_delta_x_shape)
        mask_forward = np.squeeze(weight_x.cpu().numpy())
        weight_x = weight_x * lf
        weight_x = np.squeeze(weight_x.cpu().numpy())
        score = np.sum(weight_x)
        if isWeighted:
            image = weight_x
        else:
            image = mask_forward


        if image.ndim >= 4:
            image = np.sum(image, axis=(2, 3))

        # track global limits
        global_min = min(global_min, image.min())
        global_max = max(global_max, image.max())


        ax = axes_flat[idx]
        if isScale:
            # plot with shared norm
            im = ax.imshow(image, cmap='viridis', interpolation='nearest', norm=norm)
        else:
            im = ax.imshow(image, cmap='viridis', interpolation='nearest')

        ims.append(im)
        ax.set_title(f"step = {float(step_size):.3g}\nscore = {float(score):.3g}")
        ax.axis('off')

        scores = np.append(scores, score)

    # Hide any unused subplots
    for k in range(len(step_sizes), len(axes_flat)):
        axes_flat[k].axis('off')

    # update normalization (applies to all ims)
    norm.vmin, norm.vmax = global_min, global_max

    # one shared colorbar
    fig.colorbar(ims[0], ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.04)

    fig.suptitle("Weights per step size", fontsize=12)
    plt.tight_layout()

    # plot scores
    plt.figure()
    step_sizes = step_sizes.cpu().numpy()
    sorted_indices = np.argsort(step_sizes)
    plt.plot(step_sizes[sorted_indices], scores[sorted_indices])
    plt.title("Scores Vs Step Size")

def display_single_pixel_mask_progression(finder, lf, x_idx, y_idx, kx_idx, ky_idx):
    """
    Displays the forwarded location of a single pixel in the original LF
    :param finder: the phase_mask_finder_walker obj
    :param lf: the original LF
    :param x_idx: the x index of the pixel
    :param y_idx: the y index of the pixel
    :param kx_idx: the kx index of the pixel
    :param ky_idx: the ky index of the pixel
    """
    device = torch.device("cuda")

    phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(finder.X, finder.Y, finder.SinX, finder.SinY,
                                                               finder.L)

    phase_maskx = finder.phase_maskx
    phase_masky = finder.phase_masky

    # finding the gradient angle of the phase mask
    angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                               phase_maskx, phase_masky,
                                               finder.sampling_dist_mask_plane, finder.method)

    x_min, x_max = float(torch.min(finder.Phase_X)), float(torch.max(finder.Phase_X))
    y_min, y_max = float(torch.min(finder.Phase_Y)), float(torch.max(finder.Phase_Y))

    plt.figure()
    plt.imshow(phase_maskx.cpu().numpy(), extent=(x_min, x_max, y_min, y_max))
    plt.scatter(float(phase_mask_x[x_idx, y_idx, kx_idx, ky_idx]), float(phase_mask_y[x_idx, y_idx, kx_idx, ky_idx]),
                c='r')
    plt.title(f"Phase Mask X - Interpolation Point, with Value = {angle_x1[x_idx, y_idx, kx_idx, ky_idx]}")
    plt.figure()
    plt.imshow(phase_masky.cpu().numpy(), extent=(x_min, x_max, y_min, y_max))
    plt.scatter(float(phase_mask_x[x_idx, y_idx, kx_idx, ky_idx]), float(phase_mask_y[x_idx, y_idx, kx_idx, ky_idx]),
                c='r')
    plt.title(f"Phase Mask Y - Interpolation Point, with Value = {angle_y1[x_idx, y_idx, kx_idx, ky_idx]}")

    # finding the forward locations with the delta in the angle gradient
    mask_delta_x, _ = find_forward_locations_gpu_parallel2(finder.X, finder.Y, finder.SinX,
                                                           finder.SinY, finder.L,
                                                           angle_x1, angle_y1,
                                                           torch.tensor([0], device=device))

    x_min, x_max = float(torch.min(finder.X)), float(torch.max(finder.X))
    y_min, y_max = float(torch.min(finder.Y)), float(torch.max(finder.Y))

    plt.figure()
    plt.imshow(np.sum(lf.cpu().numpy(), axis=(2, 3)), extent=(x_min, x_max, y_min, y_max))
    x_forward_locations = torch.squeeze(mask_delta_x[0])
    y_forward_locations = torch.squeeze(mask_delta_x[1])
    plt.scatter(float(x_forward_locations[x_idx, y_idx, kx_idx, ky_idx]),
                float(y_forward_locations[x_idx, y_idx, kx_idx, ky_idx]))
    plt.title("Location of Forward Warped Pixel on the Original LF")

    mask_delta_x_shape = mask_delta_x[0].shape
    flattened_shape = (mask_delta_x_shape[0], mask_delta_x_shape[1] * mask_delta_x_shape[2],
                       mask_delta_x_shape[3] * mask_delta_x_shape[4])
    mask_delta_x = (mask_delta_x[0].reshape(flattened_shape), mask_delta_x[1].reshape(flattened_shape))
    # interpolation of the mask
    mask_delta_x_loc = torch.stack(mask_delta_x, dim=-1)
    func = finder.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(1, -1, -1, -1)
    weight_x = interpolator.grid_sample(func, mask_delta_x_loc, mode=finder.method,
                                        padding_mode='zeros',
                                        align_corners=True).squeeze()

    # The cost
    weight_x = weight_x.reshape(mask_delta_x_shape)
    mask_forward = np.squeeze(weight_x.cpu().numpy())
    weight_x = weight_x * lf
    weight_x = np.squeeze(weight_x.cpu().numpy())
    # Sum over angles and plot in a subplot
    plt.figure()
    plt.imshow(np.sum(finder.mask.cpu().numpy(), axis=(2, 3)), extent=(x_min, x_max, y_min, y_max))
    plt.scatter(float(finder.X[x_idx, y_idx, kx_idx, ky_idx]), float(finder.Y[x_idx, y_idx, kx_idx, ky_idx]))
    plt.title("Pixel in Original Mask")
    plt.figure()
    plt.imshow(np.sum(mask_forward, axis=(2, 3)), extent=(x_min, x_max, y_min, y_max))
    plt.scatter(float(x_forward_locations[x_idx, y_idx, kx_idx, ky_idx]),
                float(y_forward_locations[x_idx, y_idx, kx_idx, ky_idx]))
    plt.title("Supposed Pixel in Forwarded Mask")

def display_score_for_different_deltas(finder, lf, isScale=False):
    """
    Displays the score map in the phase mask domain, for the different possible deltas
    :param finder: the phase_mask_finder_walker obj. The images' colors are coordinated
    :param lf: the original LF, for score calculation
    :param isScale: whether to scale all the image colors together
    """
    # create a list of possible x and y delta combinations
    device = torch.device("cuda")
    deltas_x = torch.linspace(-finder.max_delta, finder.max_delta, finder.n_delta, device=device)

    if 0 not in deltas_x:
        deltas_x = torch.concatenate((torch.tensor([0], device=device), deltas_x))
    else:
        # TODO: Fix this
        zero_idx = torch.where(deltas_x == 0)[0]
        deltas_x = torch.cat((torch.tensor([0], device=device), deltas_x[0:zero_idx], deltas_x[zero_idx + 1:]))

    deltas_y = deltas_x
    # finding the locations on the phase mask
    phase_mask_x, phase_mask_y = find_phase_mask_locations_gpu(finder.X, finder.Y, finder.SinX, finder.SinY, finder.L)

    # finding the gradient angle of the phase mask
    angle_x1, angle_y1 = find_mask_angles_gpu2(phase_mask_x, phase_mask_y,
                                               finder.phase_maskx, finder.phase_masky,
                                               finder.sampling_dist_mask_plane, finder.method)

    # finding the forward locations with the delta in the angle gradient
    mask_delta = find_forward_locations_gpu_parallel2_and(finder.X, finder.Y, finder.SinX, finder.SinY,
                                                          finder.L,
                                                          angle_x1, angle_y1, deltas_x, deltas_y)
    mask_delta_shape = mask_delta[0].shape
    flattened_shape = (mask_delta_shape[0], mask_delta_shape[1] * mask_delta_shape[2],
                       mask_delta_shape[3] * mask_delta_shape[4])
    mask_delta = (mask_delta[0].reshape(flattened_shape), mask_delta[1].reshape(flattened_shape))
    # interpolation of the mask
    mask_delta_loc = torch.stack(mask_delta, dim=-1)
    func = finder.mask[:, :, 0, 0].unsqueeze(0).unsqueeze(0).expand(len(deltas_x) ** 2, -1, -1, -1)
    weight = interpolator.grid_sample(func, mask_delta_loc, mode=finder.method,
                                      padding_mode='zeros',
                                      align_corners=True).squeeze()

    # The cost
    weight = weight.reshape(mask_delta_shape)
    weight = weight * lf
    ## TODO: get rid of the transpose if possible
    weight = torch.transpose(torch.flatten(weight, start_dim=1), 0, 1)

    # create transformation matrix to convert to the phase_mask shape
    transform_matrix = create_transform_matrix_gpu2(phase_mask_x, phase_mask_y,
                                                    finder.sampling_dist_mask_plane,
                                                    finder.phase_mask_shape, lf.shape)

    # update the deltas' score

    score = torch.transpose((transform_matrix @ weight), 0, 1).reshape(len(deltas_x) ** 2,
                                                                       finder.phase_mask_shape[0],
                                                                       finder.phase_mask_shape[1])


    # Prepare figure layout
    num_plots = score.shape[0]
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Shared normalization for all images
    norm = mcolors.Normalize()
    ims = []
    global_min, global_max = np.inf, -np.inf
    deltas = torch.cartesian_prod(deltas_x, deltas_y)
    for idx in range(num_plots):
        image = np.squeeze(score[idx].cpu().numpy())
        # track global limits
        global_min = min(global_min, image.min())
        global_max = max(global_max, image.max())


        ax = axes_flat[idx]
        if isScale:
            # plot with shared norm
            im = ax.imshow(image, cmap='viridis', interpolation='nearest', norm=norm)
        else:
            im = ax.imshow(image, cmap='viridis', interpolation='nearest')

        ims.append(im)
        ax.set_title(f"deltaX = {float(deltas[idx, 0]):.3g}, deltaY = {float(deltas[idx, 1]):.3g}")
        ax.axis('off')

    # Hide any unused subplots
    for k in range(num_plots, len(axes_flat)):
        axes_flat[k].axis('off')

    # update normalization (applies to all ims)
    norm.vmin, norm.vmax = global_min, global_max

    # one shared colorbar
    fig.colorbar(ims[0], ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.04)

    fig.suptitle("Weights per step size", fontsize=12)
    plt.tight_layout()

def display_phase_gradient_regions(reconstructor, lf, gradientx, gradienty, isWeighted=False):
    """
    Displays the phase mask gradient relevant to each location in the LF
    :param reconstructor: the reconstructor with which to perform the FW
    :param lf: the original LF
    :param gradientx: the phase mask gradient in the x direction
    :param gradienty: the phase mask gradient in the y direction
    :param isWeighted: whether to weight the image using the LF energy
    """
    reconstructor.lf = lf
    reconstructor.delta_sin_x = gradientx
    reconstructor.delta_sin_y = gradienty
    reconstructor.find_mask_location_points()
    delta_sin_size = reconstructor.delta_sin_x.shape
    x_delta_sin = np.linspace(0, delta_sin_size[1] * reconstructor.sampling_dist_mask_plane, delta_sin_size[1],
                              endpoint=False) - \
                  (delta_sin_size[1] - 1) * reconstructor.sampling_dist_mask_plane / 2
    y_delta_sin = np.linspace(0, delta_sin_size[0] * reconstructor.sampling_dist_mask_plane, delta_sin_size[0],
                              endpoint=False) - \
                  (delta_sin_size[0] - 1) * reconstructor.sampling_dist_mask_plane / 2

    inter1_points = np.array([reconstructor.inter1_points_x.ravel(), reconstructor.inter1_points_y.ravel()]).T
    delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), reconstructor.delta_sin_x, bounds_error=False)
    delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), reconstructor.delta_sin_y, bounds_error=False)
    imageX = delta_sin_x_func(inter1_points).reshape(reconstructor.inter1_points_x.shape)
    imageY = delta_sin_y_func(inter1_points).reshape(reconstructor.inter1_points_x.shape)

    if isWeighted:
        display_lf_2d_with_alpha(imageX, reconstructor.lf, "Delta X with Alpha")
        display_lf_2d_with_alpha(imageY, reconstructor.lf, "Delta X with Alpha")

    else:
        display_lf_2d(imageX, "Delta X")
        display_lf_2d(imageY, "Delta Y")



def display_phase_gradient_regions_from_finder(finder, lf, isWeighted=False):
    """
    Displays the phase mask gradient relevant to each location in the LF.
    This runs from the phase_mask_finder_walker

    :param finder: the phase_mask_finder_walker obj
    :param lf: the original LF
    :param isWeighted: whether to weight the image using the LF energy
    :return:
    """
    angle_finder = gradient_angle_finder(finder.sampling_dist_mask_plane, 1, finder.wavelength, finder.sigma)
    reconstructor = lf_forward_reconstructor(finder.max_sin, finder.wavelength, finder.sampling_dist_lf_plane, finder.sampling_dist_mask_plane, 1,
                                             finder.L, angle_finder,
                                             lf.shape)
    display_phase_gradient_regions(reconstructor, lf.cpu().numpy(), finder.phase_maskx.cpu().numpy(), finder.phase_masky.cpu().numpy(),
                                   isWeighted=isWeighted)
