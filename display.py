from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


def display_lf_summed(lf, name):
    image = np.sum(lf, axis=(2, 3))
    plt.figure();
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.title(f"Summed LF - {name}")
    plt.colorbar()  # Add a color bar
    #plt.show()


def display_mask(mask, name):
    plt.figure();
    plt.imshow(np.real(mask), cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a color bar
    plt.title(f"Mask - {name}")
    #plt.show()


def display(image, name):
    plt.figure();
    plt.imshow(image, cmap='viridis', interpolation='nearest')
    plt.title(name)
    plt.colorbar()  # Add a color bar
    #np.save(f'{name}.npy', image)




def display_lf_2d(lf, name):
    new_shape = (lf.shape[2] * lf.shape[0], lf.shape[3] * lf.shape[1])
    lf_2d = lf.transpose(0, 2, 1, 3).reshape(new_shape)

    max_value = np.max(lf)
    lf_2d_inv = np.pad(lf, ((1,1),(1,1),(0,0),(0,0)), constant_values=max_value)
    new_shape = (lf_2d_inv.shape[2] * lf_2d_inv.shape[0], lf_2d_inv.shape[3] * lf_2d_inv.shape[1])
    lf_2d_inv = lf_2d_inv.transpose(2, 0, 3, 1).reshape(new_shape)
    #display(lf_2d, f"2d LF - {name}")

    display(lf_2d_inv, f"2d LF Inverse - {name}")



def mask_lf(lf, radius, name, location=(0,0)):
    x, y = np.meshgrid(np.arange(lf.shape[0]), np.arange(lf.shape[0]))
    in_circle = np.power(x - location[0] - (lf.shape[0]-1) / 2, 2) + np.power(y -  location[1] - (lf.shape[1]-1) / 2, 2) < radius ** 2
    in_circle = in_circle[:, :, np.newaxis, np.newaxis]
    lf_reconstructed_mask = lf * in_circle
    display_lf_2d(lf_reconstructed_mask, name)
    display_lf_summed(lf_reconstructed_mask, name)
    return lf_reconstructed_mask

def remove_angles(lf, angles):
    for angle in angles:
        lf[:,:,angle[0],angle[1]] = 0*lf[:,:,angle[0],angle[1]];
    return lf;
