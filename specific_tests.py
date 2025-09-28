
from utils import *
from AngleFinders.gradient_angle_finder import gradient_angle_finder
from Reconstructors.lf_forward_reconstruction import lf_forward_reconstructor
from PhaseMaskFinders.phase_mask_finder import *
import cProfile
from PhaseMaskFinders.phase_mask_finder_walker import phase_mask_finder_walker
from datetime import datetime

# constant parameters
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
L = 100

# reconstruction parameters
N = 1  # window size (N*N)
phase_mask_shape = (1601, 1601)


def pure_with_actual_angle(debug):
    # run parameters
    sigma = 10
    n_deltas = 10
    #max_delta = 1 / 6
    max_delta = 0.1691
    n_stepsize = 4
    max_stepsize = 2
    n_iter = 3

    # load and display lf and mask
    lf = load_lf('pure')
    display_lf_2d(lf, "LF 2D")

    display_lf_summed(lf, 'LF')
    if debug:
        plt.show()

    # find the phase mask
    finder = phase_mask_finder_walker(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, n_deltas, max_delta, n_stepsize,
                                      max_stepsize, 'nearest')
    phase_x, phase_y = finder.find_phase_mask_gpu(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    if debug:
        plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    # TODO: Check why i need to input the phase with minus??!?
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    lf_masked = display_lf_summed_with_mask(lf_reconstructed_gradient, sampling_dist_lf_plane, sigma, "LF with MASK")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"my_array_{timestamp}.npy"
    np.save(filename, lf_masked)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    display_lf_2d(lf_reconstructed_gradient, "Reconstruced - 2D")
    plt.show()


def parabola_with_actual_angle(debug):
    # run parameters
    sigma = 5
    n_deltas = 10
    max_delta = 1 / 6
    n_stepsize = 4
    max_stepsize = 2
    n_iter = 1

    # load and display lf and mask
    lf = load_lf('parabola')
    display_lf_summed(lf, 'LF')
    if debug:
        plt.show()

    # find the phase mask
    finder = phase_mask_finder_walker(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, n_deltas, max_delta, n_stepsize,
                                      max_stepsize, 'nearest')
    phase_x, phase_y = finder.find_phase_mask_gpu(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    if debug:
        plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    # TODO: Check why i need to input the phase with minus??!?
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    lf_masked = display_lf_summed_with_mask(lf_reconstructed_gradient, sampling_dist_lf_plane, sigma, "LF with MASK")
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()


def blur_with_actual_angle(debug):
    # run parameters
    sigma = 5
    n_deltas = 10
    max_delta = 1 / 6
    n_stepsize = 4
    max_stepsize = 2
    n_iter = 10

    # load and display lf and mask
    lf = load_lf('blur40')
    display_lf_summed(lf, 'LF')
    if debug:
        plt.show()

    # find the phase mask
    finder = phase_mask_finder_walker(n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      phase_mask_shape, lf.shape, max_sin, L, n_deltas, max_delta, n_stepsize,
                                      max_stepsize, 'nearest')
    phase_x, phase_y = finder.find_phase_mask_gpu(lf)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    if debug:
        plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    # TODO: Check why i need to input the phase with minus??!?
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()


def tester(test, suffix=None, profile=False, debug=False):
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if test == 'pure with actual angle':
        pure_with_actual_angle(debug)
    elif test == 'parabola with actual angle':
        parabola_with_actual_angle(debug)
    elif test == 'blur with actual angle':
        blur_with_actual_angle(debug)
    else:
        raise NotImplemented("Test Not Implemented")
    if profile:
        profiler.disable()
        profiler.dump_stats('profile_data.prof')
