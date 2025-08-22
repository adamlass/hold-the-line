from abc import ABC, abstractmethod

from classes import Call
from classes.Dial import Action
from queue import Queue

class CallService(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def create_call(self, number: str) -> Call:
        pass
    
    @abstractmethod
    def start_call(self, call_id: str, audio_queue: Queue) -> Call:
        pass
    
    @abstractmethod
    def dial(self, call_id: str, dial: Action) -> bool:
        pass
    
    @abstractmethod
    def hangup(self, call_id: str) -> bool:
        pass
    
    
    