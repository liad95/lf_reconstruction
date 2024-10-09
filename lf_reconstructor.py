import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from abc import ABC, abstractmethod



class lf_reconstructor:
    def __init__(self, max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N, L, angle_finder,
                 lf_shape):
        self.max_sin = max_sin
        self.wavelength = wavelength
        self.sampling_dist_lf_plane = sampling_dist_lf_plane
        self.sampling_dist_mask_plane = sampling_dist_mask_plane
        self.N = N
        self.L = L
        self.angle_finder = angle_finder
        self.n_angles = lf_shape[2]
        self.height = lf_shape[0]
        self.width = lf_shape[1]

    @abstractmethod
    def reconstruct_lf(self, lf, mask):
        pass
