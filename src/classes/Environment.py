from abc import ABC, abstractmethod
from threading import Lock

from classes.Action import Action
from classes.Dial import Action
from colorama import Fore, Style
from services.TranscriptionService import TranscriptionService
from services.LLMService import LLMService
import torch
from transformers import DynamicCache

class Environment(ABC):
    id: str
    llm_service: LLMService
    transcription_service: TranscriptionService
    lock: Lock
    
    def __init__(self, id: str, llm_service: LLMService, transcription_service: TranscriptionService, log_color: str = Fore.MAGENTA):
        self.id = id
        self.name = f"Env-{id}"
        self.llm_service = llm_service
        self.transcription_service = transcription_service
        self.log_color = log_color
        self.lock = Lock()
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(f"{self.log_color}[{self.name}]:", *args, Style.RESET_ALL, flush=flush, end=end)
        
    @abstractmethod
    def start(self, *args, **kwargs):
        """
        Start the environment.
        """
        raise NotImplementedError("Start method not implemented")
    
    @abstractmethod
    def stop(self):
        """
        Stop the environment.
        """
        raise NotImplementedError("Stop method not implemented")
    
    @abstractmethod
    def apply_action(self, action: Action):
        """
        Perform an action in the environment.
        """
        raise NotImplementedError("apply_action method not implemented")
    
    @abstractmethod
    def get_user_goal(self) -> str:
        """
        Get the user goal for the environment.
        """
        raise NotImplementedError("Get user goal method not implemented")
    
    @abstractmethod
    def prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, past_key_values, cache_position: int) -> tuple[Action, DynamicCache, int]:
        raise NotImplementedError("Prompt method not implemented")