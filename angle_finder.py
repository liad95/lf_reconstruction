import numpy as np
# import imagesc
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from utils import *
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from display import *
from abc import abstractmethod


class angle_finder:
    def __init__(self):
        pass

    @abstractmethod
    def get_delta_sin(self, mask, filter=None):
        pass
