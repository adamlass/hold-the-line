# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
device = get_best_device()
print(f"Using device: {device}")
    
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

print("Model loaded successfully")

print("Moving model to device")
model.to(device)
print("Model moved to device successfully")