import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from lf_reconstructor import lf_reconstructor


class lf_backward_reconstructor(lf_reconstructor):

    def __init__(self, max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N, L, angle_finder,
                 lf_shape):
        super().__init__(max_sin, wavelength, sampling_dist_lf_plane, sampling_dist_mask_plane, N, L, angle_finder,
                         lf_shape)
        self._create_lf_grid()
    def reconstruct_lf_with_gradient(self, lf, gradientx, gradienty):
        self.lf = lf
        self.delta_sin_x = gradientx
        self.delta_sin_y = gradienty
        self._find_mask_location_points()
        self._get_interpolation_points()
        reconstructed_lf = self._interpolate_lf()
        return reconstructed_lf
    def reconstruct_lf(self, lf, mask):
        self.lf = lf
        self.mask = mask
        self.delta_sin_x, self.delta_sin_y = self.angle_finder.get_delta_sin(self.mask)
        self._find_mask_location_points()
        self._get_interpolation_points()
        reconstructed_lf = self._interpolate_lf()
        return reconstructed_lf

    def _find_mask_location_points(self):
        self.inter1_points_x = self.X + self.L * self.SinX / self.SinZ
        self.inter1_points_y = self.Y + self.L * self.SinY / self.SinZ

        # when dividing into windows

    def _get_interpolation_points(self):
        delta_sin_size = self.delta_sin_x.shape
        x_delta_sin = np.linspace(0, delta_sin_size[1] * self.sampling_dist_mask_plane, delta_sin_size[1],
                                  endpoint=False) - \
                      (delta_sin_size[1] - 1) * self.sampling_dist_mask_plane / 2
        y_delta_sin = np.linspace(0, delta_sin_size[0] * self.sampling_dist_mask_plane, delta_sin_size[0],
                                  endpoint=False) - \
                      (delta_sin_size[0] - 1) * self.sampling_dist_mask_plane / 2

        inter1_points = np.array([self.inter1_points_x.ravel(), self.inter1_points_y.ravel()]).T
        delta_sin_x_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), self.delta_sin_x, bounds_error=False)
        delta_sin_y_func = RegularGridInterpolator((x_delta_sin, y_delta_sin), self.delta_sin_y, bounds_error=False)
        delta_sin_x_interp1 = delta_sin_x_func(inter1_points).reshape(self.inter1_points_x.shape)
        delta_sin_y_interp1 = delta_sin_y_func(inter1_points).reshape(self.inter1_points_x.shape)

        # endregion

        # region part 2

        self.inter2_points_sinx = self.SinX + delta_sin_x_interp1
        self.inter2_points_siny = self.SinY + delta_sin_y_interp1
        sinz = np.sqrt(1 - np.power(self.inter2_points_sinx, 2) - np.power(self.inter2_points_siny, 2))
        self.inter2_points_x = self.X + self.L * (self.SinX / self.SinZ - self.inter2_points_sinx / sinz)
        self.inter2_points_y = self.Y + self.L * (self.SinY / self.SinZ - self.inter2_points_siny / sinz)


    def _interpolate_lf(self):
        inter2_points = np.array(
            [self.inter2_points_x.ravel(), self.inter2_points_y.ravel(), self.inter2_points_sinx.ravel(),
             self.inter2_points_siny.ravel()]).T

        lf_interp = RegularGridInterpolator((self.x, self.y, self.sin_x, self.sin_y), self.lf, bounds_error=False,
                                            fill_value=0)
        lf_reconstructed = lf_interp(inter2_points).reshape(self.inter2_points_x.shape)
        return lf_reconstructed

    def _create_lf_grid(self):
        self.sin_x = np.linspace(-self.max_sin + 1 / 14, self.max_sin - 1 / 14, self.n_angles)
        self.sin_y = np.linspace(-self.max_sin + 1 / 14, self.max_sin - 1 / 14, self.n_angles)
        self.x = np.linspace(0, self.width * self.sampling_dist_lf_plane,
                             self.width, endpoint=False) - (self.width - 1) * self.sampling_dist_lf_plane / 2
        self.y = np.linspace(0, self.height * self.sampling_dist_lf_plane,
                             self.height, endpoint=False) - (self.height - 1) * self.sampling_dist_lf_plane / 2
        self.X, self.Y, self.SinX, self.SinY = np.meshgrid(self.x, self.y, self.sin_x, self.sin_x, indexing='ij')
        self.SinZ = np.sqrt(1 - np.power(self.SinX, 2) - np.power(self.SinY, 2))

    # when dividing into windows
