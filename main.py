#from specific_tests import tester
import time

from tests import tester

start_time = time.time()
import torch
from memory_profiler import profile
torch.set_grad_enabled(False)


if __name__ == '__main__':
    # options for test : 'FW with mask', 'BW with mask', 'FW walker', 'Correlation Analysis'
    # options for suffix : 'blur40', 'blur100', 'none', 'parabola', 'pure', 'pure2', 'pure3', 'weak_parabola'

    #tester('FW with mask', 'pure')
    #tester('BW with mask', 'pure')
    #tester('FW walker', 'pure')
    tester('FW GD', 'pure')
    #tester('Correlation Analysis', 'blur100')
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")



