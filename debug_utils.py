import torch
import gc


def get_tensor_memory(tensor):
    if tensor.is_cuda:
        if isinstance(tensor, torch.sparse.Tensor) and tensor.is_sparse and tensor.layout == torch.sparse_coo:
            return get_sparse_tensor_memory(tensor)
        else:
            size_in_bytes = tensor.element_size() * tensor.nelement()
            size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
            return size_in_mb
    return 0


def get_sparse_tensor_memory(sparse_matrix):
    # Get the number of non-zero elements
    num_non_zero = sparse_matrix._nnz()

    # Get the data type of the elements
    data_type = sparse_matrix.dtype

    # Calculate the size in bytes based on the data type
    size_per_element = torch.tensor(0, dtype=data_type).element_size()

    # Total size in bytes
    total_size_bytes = num_non_zero * size_per_element

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 ** 2)

    return total_size_mb


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


class tensor_dict:
    def __init__(self):
        self.dict = {}

    def track_obj(self, obj, name):
        self.dict[id(obj)] = (name, get_tensor_memory(obj))

    # Define a function to check tensor memory

    def print_active_tensors(self):
        total_memory = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
                if id(obj) in self.dict:
                    value = self.dict[id(obj)]
                    print(f"Tensor {(value[0])}: {value[1]:.2f} MB")
                    total_memory += value[1]
        print(f"Total memory: {total_memory} MB \n")

    def update_active_tensors(self):
        total_memory = 0
        exists = dict.fromkeys(self.dict.keys(), 0)
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:  # Check if tensor is on GPU
                if id(obj) in self.dict:
                    tensor_memory = get_tensor_memory(obj)
                    self.dict[id(obj)] = (self.dict[id(obj)][0], tensor_memory)
                    total_memory += tensor_memory
                    exists[id(obj)] = 1

        for key, exist in exists.items():
            if not exist:
                del self.dict[key]