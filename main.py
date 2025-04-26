#from specific_tests import tester
from tests import tester
import numpy as np
import time
start_time = time.time()
import torch
from display import *
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

"""
'Single iter pure with actual angle'
'Double iter pure with actual angle'    
'N iter pure with actual angle'
"""

if __name__ == '__main__':

    tester('FW walker', 'pure', profile=True)
    #tester('parabola with actual angle')
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    arr1 = np.load("my_array_2024-12-17_12-58-56.npy")
    arr3 = np.load("my_array_2024-12-17_12-59-52.npy")
    """display(arr1, "Iter1")
    display(arr3, "Iter3")


    print(f"Total Score of 1 =  {np.sum(arr1)}")
    print(f"Total Score of 3 =  {np.sum(arr3)}")
    plt.show()"""


