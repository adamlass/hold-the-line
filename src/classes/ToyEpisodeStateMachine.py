import math
import random
from threading import Lock
from typing import Iterator
from classes.Action import Action
from classes.ActionType import ActionType
from classes.Episode import Episode
from classes.StateMachine import StateMachine

DATA_PATH = "data/trees"

SAMPLE_RATE = 16000                     # Hz
NUM_CHANNELS = 1                        # mono
BITS_PER_SAMPLE = 32      
SAMPLE_WIDTH = BITS_PER_SAMPLE // 8     # in bytes
CHUNK_SIZE = 1024         # bytes per yield
# S_PER_CHUNK = CHUNK_SIZE / (SAMPLE_RATE * SAMPLE_WIDTH * NUM_CHANNELS)  # duration of each chunk in ms
# print("Chunk duration:", S_PER_CHUNK, "s")
# BUFFER_SECONDS = 0.5
MAX_FRAMES = 10

class ToyEpisodeStateMachine(StateMachine):
    def __init__(self, episode: Episode):
        super().__init__(episode)
        self.expected_action_text = None
        
    def next_frame(self) -> str:
        if self.expected_action_text is None:
            rnd = random.random()
            if rnd >= 0.9:
                self.expected_action_text = "press 1"
            elif rnd >= 0.8:
                self.expected_action_text = "press 2"
        
        return "[SILENCE]" if self.expected_action_text is None else f"Please {self.expected_action_text}."
            
    def apply_action(self, action: Action) -> tuple[float, bool]:
        reward = 0.0
        
        if self.frame >= MAX_FRAMES - 1:
            self.done = True
        self.frame += 1
        
        if action.text == ActionType.wait:
            return reward, self.done
        
        if self.expected_action_text is not None:
            self.done = True
            
        if action.text == self.expected_action_text:
            return 1.0, self.done
        else:
            return -1.0, self.done
            
            