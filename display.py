from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def display_lf_summed(lf, name):
    image = np.sum(lf, axis=(2, 3))
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.title(f"Summed LF - {name}")
    plt.colorbar()  # Add a color bar
    plt.show()


def display_mask(mask, name):
    plt.imshow(np.real(mask), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.title(f"Mask - {name}")
    plt.show()


def display(image, name):
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.title(name)
    plt.colorbar()  # Add a color bar
    plt.show()

def display_lf_2d(lf, name):
    new_shape = (lf.shape[2] * lf.shape[0], lf.shape[3] * lf.shape[1])
    lf_2d_inv = lf.transpose(2, 0, 3, 1).reshape(new_shape)
    lf_2d = lf.transpose(0, 2, 1, 3).reshape(new_shape)
    display(lf_2d, f"2d LF - {name}")
    display(lf_2d_inv, f"2d LF Inverse - {name}")
