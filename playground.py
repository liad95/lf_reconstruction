import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from skimage import data, color
from display import *
from itertools import product


data = {
    (1,4): np.array([[1, 3], [2, 4]]),
    (2,5): np.array([[4, 2], [1, 5]]),
    (3,6): np.array([[2, 1], [6, 0]])
}

keys = list(data.keys())
arrays = np.array(list(data.values()))

# Stack the arrays along a new axis to create a 3D array
stacked_arrays = np.stack(arrays, axis=0)

# Create a mask where each element is True if it's the maximum along axis=0
mask = stacked_arrays == np.max(stacked_arrays, axis=0, keepdims=True)

# Use argmax on the mask along the first axis to get indices
max_indices = np.argmax(mask, axis=0)

# Map the indices back to the keys
max_key_array = np.array(keys)[max_indices]

print(max_key_array[:,:,0])
print(max_key_array[:,:,1])