from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def display_lf_summed(lf, name):
    """
    Display the LF as an image (sums over the different angles)
    :param lf: the light field
    :param name: name for the figure
    """
    image = np.sum(lf, axis=(2, 3))
    display(image, f"Summed LF - {name}")


def display_mask(mask, name):
    """
    Display the Mask as an image (displays only real part of mask)
    :param mask: the phase mask
    :param name: name for the figure
    """
    image = np.real(mask)
    display(image, f"Mask - {name}")


def display(image, name):
    """
    Displays an image
    :param image: the image to display
    :param name: name for the figure
    """
    plt.figure()
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.title(name)
    plt.colorbar()


def display_lf_2d(lf, name):
    """
    Display the LF as an image (displays all angles), where the angles are separated by lines
    :param lf: the light field
    :param name: name for the figure
    """
    max_value = np.max(lf)
    lf_2d_inv = np.pad(lf, ((1, 1), (1, 1), (0, 0), (0, 0)),
                       constant_values=max_value)  # adding the lines to separate the angles
    new_shape = (lf_2d_inv.shape[2] * lf_2d_inv.shape[0], lf_2d_inv.shape[3] * lf_2d_inv.shape[1])
    lf_2d_inv = lf_2d_inv.transpose(2, 0, 3, 1).reshape(new_shape)

    display(lf_2d_inv, f"2d LF Inverse - {name}")


def mask_lf(lf, radius, name, location=(0, 0)):
    """
    Display the LF as 2 image (displays all angles, and summed over angles), with a circular mask applied on the LF
    :param lf: the light field
    :param name: name for the figure
    """
    x, y = np.meshgrid(np.arange(lf.shape[0]), np.arange(lf.shape[0]))
    in_circle = np.power(x - location[0] - (lf.shape[0] - 1) / 2, 2) + np.power(y - location[1] - (lf.shape[1] - 1) / 2,
                                                                                2) < radius ** 2
    in_circle = in_circle[:, :, np.newaxis, np.newaxis]
    lf_reconstructed_mask = lf * in_circle
    display_lf_2d(lf_reconstructed_mask, name)
    display_lf_summed(lf_reconstructed_mask, name)
    return lf_reconstructed_mask


def remove_angles(lf, angles):
    """
    Zeros out angles in the light field
    :param lf: the light field
    :param angles: the angles to remove (list of tuples/arrays)
    :return: the LF with the zeroed angles
    """
    for angle in angles:
        lf[:, :, angle[0], angle[1]] = 0 * lf[:, :, angle[0], angle[1]];
    return lf;
