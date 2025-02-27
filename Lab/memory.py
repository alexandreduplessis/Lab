import torch

def get_memory(device):
    free, avail = torch.cuda.mem_get_info(device)
    avail /= 1e9
    free /= 1e9
    print(f"{free} / {avail}")
    return None

def tensor_memory(tensor):
    return tensor.element_size() / 1e9

def free_memory(tens_list, device):
    for tens in tens_list:
        del tens
    torch.cuda.empty_cache()
    free, avail = torch.cuda.mem_get_info(device)
    avail /= 1e9
    free /= 1e9
    print(f"{free} / {avail}")
    
def get_available_gpu(min_memory_gb=20):
    min_memory_bytes = min_memory_gb * 1024**3  # Convert GB to bytes
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        free, _ = torch.cuda.mem_get_info(device)
        if free >= min_memory_bytes:
            return device
    raise RuntimeError("No GPU with at least {} GB available memory found.".format(min_memory_gb))