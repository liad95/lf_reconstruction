from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def display_with_opacity(image, alpha, name):
    plt.figure()
    plt.imshow(image, alpha=alpha, cmap='viridis', interpolation='nearest')
    plt.title(name)
    plt.colorbar()


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


def display_with_sum(image, name):
    display(image, name + ", sum = " + str(np.sum(image)))


def display_lf_on_phase(x, y, sampling_dist, data, name):
    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
    data = data.cpu().numpy().flatten()
    x = (x + 1601 * sampling_dist / 2) / sampling_dist
    y = (y + 1601 * sampling_dist / 2) / sampling_dist
    # Plot the heatmap
    plt.figure()
    h, xedges, yedges, img = plt.hist2d(x, y, bins=(100, 50), weights=data, cmap='viridis')

    # Add a colorbar
    plt.colorbar(label='Sum of Data Value')

    # Add a colorbar

    plt.xlim(0, 1601)
    plt.ylim(1601, 0)

    # Labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(name)

    # Compute bin centers
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])  # X bin centers
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])  # Y bin centers

    # Note: `h` shape is (50, 100) because `plt.hist2d` transposes the axes.
    # The dimensions of `xcenters` and `ycenters` must align with `h.T` for interpolation.
    interpolator = RegularGridInterpolator((ycenters, xcenters), h.T, bounds_error=False, fill_value=0)

    # Create the fine grid
    grid_x = np.linspace(0, 1601, 1601)  # Fine grid along X
    grid_y = np.linspace(0, 1601, 1601)  # Fine grid along Y
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)  # Create a 2D grid
    points = np.array([grid_yy.ravel(), grid_xx.ravel()]).T  # Combine for interpolation
    alpha = interpolator(points).reshape(grid_xx.shape)  # Interpolate and reshape

    # Normalize alpha to [0, 1] for visualization
    alpha = (alpha - np.nanmin(alpha)) / (np.nanmax(alpha) - np.nanmin(alpha))
    alpha = np.nan_to_num(alpha)  # Replace NaNs with 0

    plt.figure()
    plt.imshow(alpha, cmap='viridis', interpolation='nearest')
    plt.title("alpha")

    return alpha


def display_all_patches(X, Y, sampling_dist, name):
    # Create a figure and axis
    fig, ax = plt.subplots()
    for i in range(7):
        for j in range(7):
            x = X[:, :, i, j].cpu().numpy()
            y = Y[:, :, i, j].cpu().numpy()
            x = (x + 1601 * sampling_dist / 2) / sampling_dist
            y = (y + 1601 * sampling_dist / 2) / sampling_dist
            bottom_left = (np.min(x), np.min(y))
            top_right = (np.max(x), np.max(y))
            # Calculate width and height of the rectangle
            width = top_right[0] - bottom_left[0]
            height = top_right[1] - bottom_left[1]

            # Add the rectangle
            rect = patches.Rectangle(bottom_left, width, height, linewidth=2, edgecolor='black',
                                     facecolor=(random.random(), random.random(), random.random(), 0.5))
            ax.add_patch(rect)

        # Set the limits of the plot
        ax.set_xlim(0, 1600)  # Adjust as needed
        ax.set_ylim(1600, 0)  # Adjust as needed

        # Display the plot
        plt.gca().set_aspect('equal', adjustable='box')  # Keep the rectangle aspect ratio
        plt.title(name)


def display_patch(X, Y, i, j, sampling_dist, name):
    X = X[:, :, i, j].cpu().numpy()
    Y = Y[:, :, i, j].cpu().numpy()
    X = (X + 1601 * sampling_dist / 2) / sampling_dist
    Y = (Y + 1601 * sampling_dist / 2) / sampling_dist
    bottom_left = (np.min(X), np.min(Y))
    top_right = (np.max(X), np.max(Y))
    # Calculate width and height of the rectangle
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add the rectangle
    rect = patches.Rectangle(bottom_left, width, height, linewidth=2, edgecolor='black',
                             facecolor=(random.random(), random.random(), random.random(), 0.5))
    ax.add_patch(rect)

    # Set the limits of the plot
    ax.set_xlim(0, 1600)  # Adjust as needed
    ax.set_ylim(1600, 0)  # Adjust as needed

    # Display the plot
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the rectangle aspect ratio
    plt.title(name)


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
