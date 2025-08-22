import os
import time
from classes.Environment import Environment
from services.TranscriptionService import TranscriptionService
from transceivers.Transceiver import Transceiver
from colorama import Fore
import logging
import sys
from scripts.whisper_online import ASRBase, FasterWhisperASR, MLXWhisper, VACOnlineASRProcessor
import numpy as np
from dotenv import load_dotenv
from utils import get_best_device
from threading import Lock
load_dotenv(override=True)


SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

SOURCE_LAN = "en"  # source language
TARGET_LAN = "en"  # target language  -- same as source for ASR, "en" if translate task is used
MIN_CHUNK_SIZE = 1.0
BUFFER_TRIMMING = "segment"
BUFFER_TRIMMING_SEC = 15
MODEL_NAME = "tiny.en"
LOGFILE=sys.stderr
TARGET_SAMPLES = int(MIN_CHUNK_SIZE * SAMPLE_RATE)

SAMPLE_RATE = 16000       # Hz
NUM_CHANNELS = 1          # mono
BITS_PER_SAMPLE = 32      
SAMPLE_WIDTH = BITS_PER_SAMPLE // 8  # in bytes
CHUNK_SIZE = 1024   
S_PER_CHUNK = CHUNK_SIZE / (SAMPLE_RATE * SAMPLE_WIDTH * NUM_CHANNELS)

SILENCE_THRESHOLD = 10
SILENCE_CHUNKS_THRESHOLD = int(SILENCE_THRESHOLD / S_PER_CHUNK)
print("Silence chunks:", SILENCE_CHUNKS_THRESHOLD)

class Transcriber(Transceiver):
    lock: Lock = Lock()
    
    silence_chunks: float
    
    def __init__(self, environment: Environment):
        super().__init__("Transcriber", environment, log_color=Fore.WHITE, wait_for_input=True, buffer_size=1, wait_for_output=True)
        self.silence_chunks = 0
        self.transcription_service = environment.transcription_service
    
    def setup(self, ):
        self.log("Setting up")
        self.online = VACOnlineASRProcessor(MIN_CHUNK_SIZE,
                                            self.transcription_service.asr_backend,
                                            tokenizer=None,
                                            logfile=LOGFILE,
                                            buffer_trimming=(BUFFER_TRIMMING, BUFFER_TRIMMING_SEC))
        
    def teardown(self):
        self.log("Tearing down")
        with self.lock:
            o = self.online.finish()
        self.log("Final output:", o)
    
    def process(self, data):
        try:
            self.online.insert_audio_chunk(data)

            with self.lock:
                o = self.online.process_iter()
            
            if o[0] is not None:
                self.silence_chunks = 0
                return o[2]
            else:
                self.silence_chunks += 1
                if self.silence_chunks >= SILENCE_CHUNKS_THRESHOLD:
                    self.silence_chunks = 0
                    return "[SILENCE]"
                    
        except AssertionError as e:
            self.log(f"assertion error: {e}")
        
        return None