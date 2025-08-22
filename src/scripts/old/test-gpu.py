import torch
import os
import multiprocessing

def worker(gpu_id):
    """Function that runs on a specific GPU"""
    torch.cuda.set_device(gpu_id)  # Set the GPU for this process
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Process {gpu_id} using {device}")

    # Example: Create a tensor on this GPU
    x = torch.rand(5, 5).to(device)
    print(f"Tensor on {device}: {x}")

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()  # Get number of GPUs allocated by SLURM
    print(f"Number of available GPUs: {num_gpus}")

    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()