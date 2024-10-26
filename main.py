from tests import tester
import time
start_time = time.time()
import torch


torch.set_grad_enabled(False)


if __name__ == '__main__':

    tester('FW walker', 'blur40', profile=True)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")