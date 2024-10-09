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


def load_lf_data():
    # Load the .mat file
    lf = loadmat('lf.mat')['I_LF']
    mask = loadmat('mask.mat')['maskL']
    N_mask = int(np.sqrt(len(mask)))
    mask = np.reshape(mask, (N_mask, N_mask))
    # mask = np.transpose(mask)

    return lf, mask
def load_gradients():
    gradienty = loadmat('gradient.mat')['gradientx']
    gradientx = loadmat('gradient.mat')['gradienty']
    return gradientx, gradienty

# params
max_sin = 0.5
wavelength = 0.5
sampling_dist_lf_plane = wavelength / 2
sampling_dist_mask_plane = wavelength / 2
sigma = 1
N = 1  # window size (N*N)
L = 100

# loading lf and mask, and extracting params
lf, mask = load_lf_data()
#lf[:,:,3,1] = 0
"""gradientx, gradienty = load_gradients()
gradientx[0,:] = 0
gradientx[1600,:] = 0
gradienty[0,:] = 0
gradienty[1600,:] = 0
gradientx[:,0] = 0
gradientx[:,1600] = 0
gradienty[:,0] = 0
gradienty[:,1600] = 0
"""


display_mask(mask, "Original")
display_lf_summed(lf, "Origina"
                      "l")
lf_size = (lf.shape[0], lf.shape[1])
n_angles = lf.shape[2]
mode = "forward"
angle_finder = gradient_angle_finder(sampling_dist_mask_plane, N, wavelength, sigma)

if mode == "backward":

    reconstructor = lf_backward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                              L, angle_finder,
                                              lf.shape)
else:
    reconstructor = lf_forward_reconstructor(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N,
                                             L, angle_finder,
                                             lf.shape)
lf_reconstructed = reconstructor.reconstruct_lf(lf, mask)
"""display(gradientx, "gradienx")
display(gradienty, "gradieny")
lf_reconstructed_gradient = reconstructor.reconstruct_lf_with_gradient(lf, gradientx, gradienty)"""

#print(np.unravel_index(np.argmax(lf_reconstructed[:, :, 0, 2]),shape=lf_reconstructed[:, :, 0, 2].shape))
display_lf_summed(lf_reconstructed, "Reconstructed")
#display_lf_summed(lf_reconstructed_gradient, "Gradient Reconstructed")
display_lf_summed(lf, "Original")
#display_lf_summed(lf_reconstructed - lf_reconstructed_gradient, "Reconstructed Diff")

#display_lf_2d(lf_reconstructed, f"Reconstructed {mode} No - (3,1)")

"""angles = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
          (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]"""

"""max_idx_lf = {}
max_idx_lf_reconstructed = {}
for i in range(lf.shape[2]):
    for j in range(lf.shape[3]):
        max_idx_lf[(i, j)] = np.unravel_index(np.argmax(lf[:, :, i, j]), shape=(lf.shape[0], lf.shape[1]))
        max_idx_lf_reconstructed[(i, j)] = np.unravel_index(np.argmax(lf_reconstructed[:, :, i, j]),
                                                            shape=(
                                                                lf_reconstructed.shape[0], lf_reconstructed.shape[1]))

mask_lf(lf_reconstructed, 10, "lf reconstructed mask")

print(max_idx_lf)
print(max_idx_lf_reconstructed)
print(f"The maximum is at {np.unravel_index(np.argmax(lf), shape=lf.shape)}")
print(f"The maximum is at {np.unravel_index(np.argmax(lf_reconstructed), shape=lf_reconstructed.shape)}")
print(
    f"The maximum is at {np.unravel_index(np.argmax(np.sum(lf_reconstructed, axis=(2, 3))), shape=(lf_reconstructed.shape[0], lf_reconstructed.shape[1]))}")



"""
plt.show()