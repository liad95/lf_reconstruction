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

# params
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
sigma = 1
N = 1  # window size (N*N)
L = 100
n_deltas = 5
max_delta = 1 / 6
# loading lf and mask
lf = load_lf("pure")


def forward_reconstruct_based_on_given_phase_mask(suffix, debug=False):
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


def backward_reconstruct_based_on_given_phase_mask(suffix, debug=False):
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


def forward_reconstruct_with_walker(suffix, debug):
    # load and display lf and mask
    lf = load_lf(suffix)
    display_lf_summed(lf, 'LF')
    plt.show()

    # find the phase mask
    finder = phase_mask_finder_walker(1, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma,
                                      (1601, 1601), lf.shape, max_sin, L, n_deltas, max_delta, 'nearest')
    phase_x, phase_y = finder.find_phase_mask(lf)
    phase_x = cp.asnumpy(phase_x)
    phase_y = cp.asnumpy(phase_y)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()


def forward_reconstruct_with_gd(suffix, debug):
    # load and display lf and mask
    lf = load_lf(suffix)
    display_lf_summed(lf, 'LF')
    plt.show()

    # find the phase mask
    finder = phase_mask_finder_gd(1, sampling_dist_mask_plane, sampling_dist_lf_plane, wavelength, sigma, (1601, 1601),
                                  lf.shape, max_sin, L, 0.1, 0.01)
    phase_x, phase_y = finder.find_phase_mask(lf)
    phase_x = cp.asnumpy(phase_x)
    phase_y = cp.asnumpy(phase_y)
    display(phase_x, "angle x")
    display(phase_y, "angle y")
    plt.show()

    # FW reconstruction using the phase mask
    angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
    lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
    display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
    plt.show()


def tester(test, suffix, profile=False, debug=False):
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if test == 'FW with mask':
        forward_reconstruct_based_on_given_phase_mask(suffix, debug)
    elif test == 'BW with mask':
        backward_reconstruct_based_on_given_phase_mask(suffix, debug)
    elif test == 'FW walker':
        forward_reconstruct_with_walker(suffix, debug)
    elif test == 'FW gd':
        forward_reconstruct_with_gd(suffix, debug)

    if profile:
        profiler.disable()
        profiler.dump_stats('profile_data.prof')
