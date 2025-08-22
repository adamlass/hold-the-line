
from abc import ABC

import numpy as np
from scripts.whisper_online import ASRBase, FasterWhisperASR, MLXWhisper
from utils import get_best_device

SOURCE_LAN = "en"  # source language
TARGET_LAN = "en"  # target language  -- same as source for ASR, "en" if translate task is used
MODEL_NAME = "tiny.en"

SAMPLE_RATE = 16000
MIN_CHUNK_SIZE = 1.0
TARGET_SAMPLES = int(MIN_CHUNK_SIZE * SAMPLE_RATE)

class TranscriptionService(ABC):
    asr_backend: ASRBase
    
    def __init__(self):
        device = get_best_device()
        
        if device == "mps":
            self.asr_backend = MLXWhisper(SOURCE_LAN, MODEL_NAME)
        else:
            self.asr_backend = FasterWhisperASR(SOURCE_LAN, MODEL_NAME)

        print("Warming up ASR backend")
        self.asr_backend.transcribe(np.zeros(TARGET_SAMPLES))
        