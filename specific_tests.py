from memory_profiler import memory_usage
import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from scipy.io import savemat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from gradient_angle_finder import gradient_angle_finder
from lf_backward_reconstructor import lf_backward_reconstructor
from lf_forward_reconstruction import lf_forward_reconstructor
import cupy as cp
from phase_mask_finder import *
import cProfile
from phase_mask_finder_gd import phase_mask_finder_gd
from phase_mask_finder_walker import phase_mask_finder_walker


# constant parameters
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
L = 100

# reconstruction parameters
N = 1  # window size (N*N)
phase_mask_shape = (1601, 1601)


def single_iter_pure_with_actual_angle(debug):
    # run parameters
    sigma = 5
    n_deltas = 30
    max_delta = 1 / 6
    n_stepsize = 10
    max_stepsize = 2
    n_iter = 1


    # load and display lf and mask
    lf = load_lf('pure')
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
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, -phase_x, -phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()

def double_iter_pure_with_actual_angle(debug):
    # run parameters
    sigma = 2
    n_deltas = 30
    max_delta = 1 / 6
    n_stepsize = 10
    max_stepsize = 2
    n_iter = 2


    # load and display lf and mask
    lf = load_lf('pure')
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
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()

def N_iter_pure_with_actual_angle(debug):
    # run parameters
    sigma = 2
    n_deltas = 30
    max_delta = 1 / 6
    n_stepsize = 10
    max_stepsize = 2
    n_iter = 10


    # load and display lf and mask
    lf = load_lf('pure')
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
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()



def tester(test, suffix=None, profile=False, debug=False):
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if test == 'Single iter pure with actual angle':
        single_iter_pure_with_actual_angle(debug)
    elif test == 'Double iter pure with actual angle':
        double_iter_pure_with_actual_angle(debug)
    elif test == 'N iter pure with actual angle':
        N_iter_pure_with_actual_angle(debug)
    else:
        raise NotImplemented("Test Not Implemented")
    if profile:
        profiler.disable()
        profiler.dump_stats('profile_data.prof')
