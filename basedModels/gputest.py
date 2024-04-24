import torch


if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")
else:
    print("CUDA (GPU) is not available.")
