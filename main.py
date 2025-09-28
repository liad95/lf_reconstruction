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

    #tester('FW walker', 'blur40', profile=True)
    tester('FW with mask', 'blur100', profile=True)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")



