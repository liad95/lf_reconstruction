import matplotlib.pyplot as plt
from memory_profiler import profile
from utils import *
from AngleFinders.gradient_angle_finder import gradient_angle_finder
from Reconstructors.lf_backward_reconstructor import lf_backward_reconstructor
from Reconstructors.lf_forward_reconstruction import lf_forward_reconstructor
from PhaseMaskFinders.phase_mask_finder import *
import cProfile
from PhaseMaskFinders.phase_mask_finder_walker import phase_mask_finder_walker
from PhaseMaskFinders.phase_mask_finder_gd import phase_mask_finder_gd
from debug_utils import *

# params
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
sigma = 2
N = 1  # window size (N*N)
L = 100
n_deltas = 5
max_delta = 1 / 6


# region general

def forward_reconstruct_based_on_given_phase_mask(suffix, calcCorr):
    # load and display lf and mask
    lf, mask = load_lf_mask(suffix)
    display_lf_summed(lf, 'LF')
    display_mask(mask, 'Mask')

    # perform forward warping
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    reconstructed_lf = reconstructor.reconstruct_lf(lf, mask)

    # display recpnstructed LF
    display_lf_summed(reconstructed_lf, 'FW Reconstructed LF')
    plt.show()

    correlation, correlation_with_angles, energy_ratio = 0, 0, 0
    if calcCorr:
        lf, mask = load_lf_mask('none')
        correlation = calc_correlation(np.sum(lf, axis=(2, 3)), np.sum(reconstructed_lf, axis=(2, 3)))
        correlation_with_angles = calc_correlation(lf, reconstructed_lf)
        energy_ratio = np.sum(reconstructed_lf*reconstructed_lf)/np.sum(lf*lf)
    return correlation, correlation_with_angles, energy_ratio


def backward_reconstruct_based_on_given_phase_mask(suffix, calcCorr):
    # load and display lf and mask
    lf, mask = load_lf_mask(suffix)
    display_lf_summed(lf, 'LF')
    display_mask(mask, 'Mask')

    # perform backward warping
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_backward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                              L, angle_finder, lf.shape)
    reconstructed_lf = reconstructor.reconstruct_lf(lf, mask)

    # display reconstructed LF
    display_lf_summed(reconstructed_lf, 'BW Reconstructed LF')
    plt.show()

    correlation, correlation_with_angles, energy_ratio = 0, 0, 0
    if calcCorr:
        lf, mask = load_lf_mask('none')
        correlation = calc_correlation(np.sum(lf, axis=(2, 3)), np.sum(reconstructed_lf, axis=(2, 3)))
        correlation_with_angles = calc_correlation(lf, reconstructed_lf)
        energy_ratio = np.sum(reconstructed_lf * reconstructed_lf) / np.sum(lf * lf)
    return correlation, correlation_with_angles, energy_ratio

def forward_reconstruct_with_walker(suffix):
    # load and display lf and mask
    lf = load_lf(suffix)
    display_lf_summed(lf, 'LF')
    n_stepsize = 10
    max_stepsize = 2
    n_iter = 2
    phase_mask_shape = (1601, 1601)

    # find the phase mask
    finder = phase_mask_finder_walker(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, n_deltas, max_delta, n_stepsize,
                                      max_stepsize, 'nearest')
    phase_x, phase_y = finder.find_phase_mask_gpu(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    # find_phase_mask_loc_relevant_to_recon_point(142, 17, 2, 4, 0.5, lf_reconstructed_gradient, sampling_dist_lf_plane,
    #                                             100, phase_x, phase_y)
    x, y = np.unravel_index(np.argmax(np.squeeze(np.sum(lf, axis=(2, 3)))), lf.shape[0:2])
    # display_forwarded_loc(x, y, max_sin, lf, sampling_dist_lf_plane, L, phase_x, phase_y, sampling_dist_mask_plane)
    # display_phase_gradient_regions(reconstructor, lf, phase_x, phase_y)
    plt.show()


def forward_reconstruct_with_gd(suffix):
    # load and display lf and mask
    lf = load_lf(suffix)
    display_lf_summed(lf, 'LF')
    stepsize = 1
    delta = 1/6
    n_iter = 2
    phase_mask_shape = (1601, 1601)

    # find the phase mask
    finder = phase_mask_finder_gd(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, stepsize, delta)
    phase_x, phase_y = finder.find_phase_mask(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()



def correlation_analysis():
    suffixes = ["none", "pure", "parabola", "blur40", "blur100"]
    fw_correlation = np.zeros(len(suffixes))
    fw_correlation_with_angles = np.zeros(len(suffixes))
    fw_energy_ratio = np.zeros(len(suffixes))
    bw_correlation = np.zeros(len(suffixes))
    bw_correlation_with_angles = np.zeros(len(suffixes))
    bw_energy_ratio = np.zeros(len(suffixes))
    for suffix_idx, suffix in enumerate(suffixes):
        correlation, correlation_with_angles, energy_ratio = forward_reconstruct_based_on_given_phase_mask(suffix=suffix, calcCorr=True)
        fw_correlation[suffix_idx] = correlation
        fw_correlation_with_angles[suffix_idx] = correlation_with_angles
        fw_energy_ratio[suffix_idx] = energy_ratio
        correlation, correlation_with_angles, energy_ratio = backward_reconstruct_based_on_given_phase_mask(suffix=suffix, calcCorr=True)
        bw_correlation[suffix_idx] = correlation
        bw_correlation_with_angles[suffix_idx] = correlation_with_angles
        bw_energy_ratio[suffix_idx] = energy_ratio

    # Extract values in consistent order

    x = 0.5*np.arange(len(suffixes))  # group positions
    width = 0.1  # bar width

    plt.figure(figsize=(10, 5))

    plt.bar(x - 1.5 * width, fw_correlation, width, label="FW")
    plt.bar(x - 0.5 * width, bw_correlation, width, label="BW")
    plt.bar(x + 0.5 * width, fw_correlation_with_angles, width, label="FW with Angles")
    plt.bar(x + 1.5 * width, bw_correlation_with_angles, width, label="BW with Angles")

    plt.xticks(x, suffixes)
    plt.ylabel("Correlation")
    plt.xlabel("Phase Mask")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.legend(loc="lower right", framealpha=1)

    x = 0.25 * np.arange(len(suffixes))
    plt.figure(figsize=(10, 5))

    plt.bar(x - 1.5 * width, fw_energy_ratio, width, label="FW")
    plt.bar(x - 0.5 * width, bw_energy_ratio, width, label="BW")

    plt.xticks(x, suffixes)
    plt.ylabel("LF Energy Ratio")
    plt.xlabel("Phase Mask")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.legend(loc="lower right", framealpha=1)
    plt.show()






# endregion

# region specific

# endregion





def tester(test, suffix=None, calcCorr=False, profile=False):
    correlation, correlation_with_angles = 0, 0
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if test == 'FW with mask':
        forward_reconstruct_based_on_given_phase_mask(suffix, calcCorr)
    elif test == 'BW with mask':
        backward_reconstruct_based_on_given_phase_mask(suffix, calcCorr)
    elif test == 'FW walker':
        forward_reconstruct_with_walker(suffix)
    elif test == 'FW GD':
        forward_reconstruct_with_gd(suffix)
    elif test == 'Correlation Analysis':
        correlation_analysis()
    else:
        raise NotImplemented("Test Not Implemented")
    if profile:
        profiler.disable()
        profiler.dump_stats('profile_data.prof')
