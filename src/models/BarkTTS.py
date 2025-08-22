import torch
from transformers import AutoProcessor, BarkModel
from utils import get_best_device
import soundfile as sf
import numpy as np

DEFAULT_VOICE_PRESET = "v2/en_speaker_6"

class BarkTTS:
    def __init__(self):
        print("Starting BarkTTS")
        self.device = get_best_device()
        print("using device:", self.device)
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16)
        print("Model loaded")
        print("Sending model to device:", self.device)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        print("... model sent to device:", self.device)
        # self.model.enable_cpu_offload()
        self.sampling_rate = self.model.generation_config.sample_rate
    
    def generate_audio(self, text, voice_preset=DEFAULT_VOICE_PRESET):
        print("Generating auido")
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        audio_array = self.model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        return audio_array
    
    def save_audio(self, audio, file_path):
        print("Saving audio to location:",file_path)
        audio2 = (audio * self.sampling_rate).astype(np.int16)
        sf.write(file_path, audio2, samplerate=self.sampling_rate)
    
if __name__ == "__main__":
    texts = [
       "Invalid option. Please try again."
    ]
    tts = BarkTTS()
    
    for i in range(len(texts)):
        file_path = f"data/test{i}.wav"
        audio = tts.generate_audio(texts[i])
        tts.save_audio(audio, file_path)
    