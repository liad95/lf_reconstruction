import torch
import time
import gc

obj_dict = {}


def get_gpu_memory_status():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()

        # Get total, allocated, and reserved memory
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        memory_allocated = torch.cuda.memory_allocated(gpu_id)
        memory_reserved = torch.cuda.memory_reserved(gpu_id)

        # Format the output to return memory in GB

        print(f"total_memory: {total_memory / (1024 ** 3)}")
        print(f"memory_allocated: {memory_allocated / (1024 ** 3)}")
        print(f"memory_reserved: {memory_reserved / (1024 ** 3)}")
    else:
        raise RuntimeError("CUDA is not available on this system.")


def track_obj(obj, name):
    obj_dict[id(obj)] = (name, get_tensor_memory(obj))


# Define a function to check tensor memory
def get_tensor_memory(tensor):
    if tensor.is_cuda:
        size_in_bytes = tensor.element_size() * tensor.nelement()
        size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
        return size_in_mb
    return 0


def print_active_tensors():
    total_memory = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
            if id(obj) in obj_dict:
                value = obj_dict[id(obj)]
                print(f"Tensor {(value[0])}: {value[1]} MB")
                total_memory += value[1]
    print(f"Total memory: {total_memory} MB")


def update_active_tensors():
    total_memory = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
            if id(obj) in obj_dict:
                tensor_memory = get_tensor_memory(obj)
                obj_dict[id(obj)] = (obj_dict[id(obj)][0], tensor_memory)
                total_memory += tensor_memory


def func(i):
    device = torch.device("cuda:0")
    size = 2500  # Large size to fill up memory and load the GPU
    matrix1 = torch.randn(size, size, device=device)
    track_obj(matrix1, "matrix1-" + str(i))
    matrix2 = torch.randn(size, size, device=device)
    track_obj(matrix2, "matrix2-" + str(i))
    return matrix1 @ matrix2


def func2(matrix1, matrix2, i):
    device = torch.device("cuda:0")
    matrix1_exp = torch.exp(matrix1)
    matrix2_exp = torch.exp(matrix2)
    track_obj(matrix1_exp, "matrix1_exp-" + str(i))
    track_obj(matrix2_exp, "matrix2_exp-" + str(i))
    return matrix1_exp @ matrix2_exp


def test4():
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA GPU is not available.")
    else:
        # Set device to GPU 0
        device = torch.device("cuda:0")
        sums = {}
        size = 2500
        matrix1 = torch.randn(size, size, device=device)
        matrix2 = torch.randn(size, size, device=device)
        track_obj(matrix1, "matrix1")
        track_obj(matrix2, "matrix2")

        # Specify matrix size to load the GPU heavily

        for i in range(10):
            result = func2(matrix1, matrix2, i)
            track_obj(result, "result-" + str(i))
            sums[i] = result
            torch.cuda.synchronize()  # Ensure all operations have finished
            gc.collect()
            update_active_tensors()
            print_active_tensors()
            print(f"Matrix multiplication completed\n")


def test3():
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA GPU is not available.")
    else:
        # Set device to GPU 0
        device = torch.device("cuda:0")
        sums = {}

        # Specify matrix size to load the GPU heavily

        for i in range(10):
            device = torch.device("cuda:0")
            size = 2500  # Large size to fill up memory and load the GPU
            matrix1 = torch.randn(size, size, device=device)
            track_obj(matrix1, "matrix1-" + str(i))
            matrix2 = torch.randn(size, size, device=device)
            track_obj(matrix2, "matrix2-" + str(i))
            result = matrix1 @ matrix2
            track_obj(result, "result-" + str(i))
            sums[i] = result
            torch.cuda.synchronize()  # Ensure all operations have finished
            gc.collect()
            update_active_tensors()
            print_active_tensors()
            print(f"Matrix multiplication completed\n")


def test2():
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA GPU is not available.")
    else:
        # Set device to GPU 0
        device = torch.device("cuda:0")
        sums = {}

        # Specify matrix size to load the GPU heavily

        for i in range(10):
            result = func(i)
            track_obj(result, "result-" + str(i))
            sums[i] = result
            torch.cuda.synchronize()  # Ensure all operations have finished
            gc.collect()
            update_active_tensors()
            print_active_tensors()
            print(f"Matrix multiplication completed\n")


def test():
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("CUDA GPU is not available.")
    else:
        # Set device to GPU 0
        device = torch.device("cuda:0")

        # Specify matrix size to load the GPU heavily
        size = 2500  # Large size to fill up memory and load the GPU
        for i in range(10):
            # Create two large random matrices on the GPU
            print("Generating large matrices...")
            matrix1 = torch.randn(size, size, device=device)
            track_obj(matrix1, "matrix1-" + str(i))
            matrix2 = torch.randn(size, size, device=device)
            track_obj(matrix2, "matrix2-" + str(i))
            update_active_tensors()
            print_active_tensors()
            get_gpu_memory_status()

            # Perform matrix multiplication and time it
            print("Performing matrix multiplication on the GPU...")
            start_time = time.time()
            result = torch.mm(matrix1, matrix2)
            track_obj(result, "result-" + str(i))
            torch.cuda.synchronize()  # Ensure all operations have finished
            end_time = time.time()
            """del matrix1
            del matrix2
            del result
            torch.cuda.empty_cache()  # Clear cached memory
            gc.collect()"""

            get_gpu_memory_status()
            # Report the time taken
            print(f"Matrix multiplication completed. Time taken: {end_time - start_time:.2f} seconds \n")


test4()
# Keep the program running so you can check GPU usage in the Task Manager
# input("Check your task manager for GPU usage, then press Enter to exit.")
