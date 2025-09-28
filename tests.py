from utils import *
from AngleFinders.gradient_angle_finder import gradient_angle_finder
from Reconstructors.lf_backward_reconstructor import lf_backward_reconstructor
from Reconstructors.lf_forward_reconstruction import lf_forward_reconstructor
from PhaseMaskFinders.phase_mask_finder import *
import cProfile
from PhaseMaskFinders.phase_mask_finder_walker import phase_mask_finder_walker

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
def forward_reconstruct_based_on_given_phase_mask(suffix):
    # load and display lf and mask
    lf, mask = load_lf_mask(suffix)
    display_lf_summed(lf, 'LF')
    display_mask(mask, 'Mask')
    plt.show()

    # perform forward warping
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    reconstructed_lf = reconstructor.reconstruct_lf(lf, mask)

    # display recpnstructed LF
    display_lf_summed(reconstructed_lf, 'FW Reconstructed LF')
    plt.show()


def backward_reconstruct_based_on_given_phase_mask(suffix):
    # load and display lf and mask
    lf, mask = load_lf_mask(suffix)
    display_lf_summed(lf, 'LF')
    display_mask(mask, 'Mask')
    plt.show()

    # perform backward warping
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_backward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                              L, angle_finder, lf.shape)
    reconstructed_lf = reconstructor.reconstruct_lf(lf, mask)

    # display reconstructed LF
    display_lf_summed(reconstructed_lf, 'BW Reconstructed LF')
    plt.show()


def forward_reconstruct_with_walker(suffix):
    # load and display lf and mask
    lf = load_lf(suffix)
    display_lf_summed(lf, 'LF')
    #plt.show()
    n_stepsize = 10
    max_stepsize = 2
    n_iter = 1
    phase_mask_shape = (1601, 1601)

    # find the phase mask
    finder = phase_mask_finder_walker(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, n_deltas, max_delta, n_stepsize,
                                      max_stepsize, 'nearest')
    phase_x, phase_y = finder.find_phase_mask_gpu(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    #plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, -phase_x, -phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()

# endregion

# region specific

# endregion



def tester(test, suffix=None, profile=False):
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if test == 'FW with mask':
        forward_reconstruct_based_on_given_phase_mask(suffix)
    elif test == 'BW with mask':
        backward_reconstruct_based_on_given_phase_mask(suffix)
    elif test == 'FW walker':
        forward_reconstruct_with_walker(suffix)
    else:
        raise NotImplemented("Test Not Implemented")
    if profile:
        profiler.disable()
        profiler.dump_stats('profile_data.prof')
