import numpy as np
import scipy
from scipy.sparse import linalg
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from lf_reconstructor import lf_reconstructor
import math
import cupy as cp
from cupyx.scipy.interpolate import RegularGridInterpolator as RegularGridInterpolatorGPU
from joblib import Parallel, delayed
from itertools import product
import dask
from dask import delayed
from cupyx.scipy.sparse import csr_matrix as csr_gpu


class phase_mask_finder:

    def __init__(self, n_iter, sampling_dist_mask_plane, sampling_dist_lf_plane, N, wavelength, sigma, phase_mask_shape,
                 lf_shape, max_sin, L):
        self.n_iter = n_iter
        self.sampling_dist_mask_plane = sampling_dist_mask_plane
        self.sampling_dist_lf_plane = sampling_dist_lf_plane
        self.N = N
        self.wavelength = wavelength
        self.sigma = sigma
        self.lf_shape = lf_shape
        self.phase_mask_shape = phase_mask_shape
        self.max_sin = max_sin
        self.L = L
        self._create_grids()
        self._create_phase_mask()
        self._create_mask()

    def find_phase_mask(self, lf):
        raise NotImplemented("Find Phase Mask Not Implemented")

    def _create_grids(self):
        # lf grid
        sin_x = np.linspace(-self.max_sin+1/14, self.max_sin-1/14, self.lf_shape[3])
        sin_y = np.linspace(-self.max_sin+1/14, self.max_sin-1/14, self.lf_shape[2])
        x = np.linspace(0, self.lf_shape[1] * self.sampling_dist_lf_plane,
                        self.lf_shape[1], endpoint=False) - (self.lf_shape[1] - 1) * self.sampling_dist_lf_plane / 2
        y = np.linspace(0, self.lf_shape[0] * self.sampling_dist_lf_plane,
                        self.lf_shape[0], endpoint=False) - (self.lf_shape[0] - 1) * self.sampling_dist_lf_plane / 2
        self.X, self.Y, self.SinX, self.SinY = np.meshgrid(x, y, sin_x, sin_y, indexing='ij')
        self.SinZ = np.sqrt(1 - np.power(self.SinX, 2) - np.power(self.SinY, 2))
        # phase mask grid

        x = np.linspace(0, self.phase_mask_shape[1] * self.sampling_dist_mask_plane,
                        self.phase_mask_shape[1], endpoint=False) - (
                    self.phase_mask_shape[1] - 1) * self.sampling_dist_mask_plane / 2
        y = np.linspace(0, self.phase_mask_shape[0] * self.sampling_dist_mask_plane,
                        self.phase_mask_shape[0], endpoint=False) - (
                    self.phase_mask_shape[0] - 1) * self.sampling_dist_mask_plane / 2
        self.Phase_X, self.Phase_Y = np.meshgrid(x, y, indexing='ij')
        pass

    def _create_mask(self):
        # self.R = 10
        self.sigma = 10
        self.mask = np.exp(-(np.power(self.X, 2) + np.power(self.Y, 2)) / (2 * self.sigma))
        display_lf_summed(self.mask, "Mask")
        plt.show()

    def _create_phase_mask(self):
        """self.phase_maskx = np.random.uniform(-self.max_sin, self.max_sin, size=self.phase_mask_shape)
        self.phase_masky = np.random.uniform(-self.max_sin, self.max_sin, size=self.phase_mask_shape)"""
        self.phase_maskx = np.zeros(self.phase_mask_shape)
        self.phase_masky = np.zeros(self.phase_mask_shape)
