
import uuid
import torch

def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def generate_unique_id():
    return uuid.uuid4().hex