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
max_delta = 1/6
profile = False
# loading lf and mask
lf = load_lf("pure")


display_lf_summed(lf, "LF")
phase_mask_finder1 = phase_mask_finder_walker(1, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma,
                                         (1601, 1601),
                                        lf.shape, max_sin, L, n_deltas, max_delta, 'nearest')
if profile :
    profiler = cProfile.Profile()
    profiler.enable()

# Run the function you want to profile
phase_x, phase_y = phase_mask_finder1.find_phase_mask(lf)
phase_x = cp.asnumpy(phase_x)
phase_y = cp.asnumpy(phase_y)
display(phase_x, "angle x")
display(phase_y, "angle y")

# Disable the profiler and save the results to a file
if profile :
    profiler.disable()
    profiler.dump_stats('profile_data.prof')



plt.show()

angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)
reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, phase_x, phase_y)
display_lf_summed(lf_reconstructed_gradient, "Reconstructed")
plt.show()