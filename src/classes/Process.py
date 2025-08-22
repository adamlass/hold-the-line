
from abc import ABC

from classes.EpisodeEnvironment import EpisodeEnvironment
from colorama import Fore, Style
from services.LLMService import LLMService
from services.TranscriptionService import TranscriptionService

PROCESS_COLORS = [Fore.LIGHTMAGENTA_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTWHITE_EX]

class Process(ABC):
    current_env: EpisodeEnvironment
    
    def __init__(self, index: int, llm_service: LLMService, transcription_service: TranscriptionService):
        self.index = index
        self.name = f"P{index}"
        self.llm_service = llm_service
        self.transcription_service = transcription_service
        self.log_color = PROCESS_COLORS[index % len(PROCESS_COLORS)]
        self.current_env = None
        
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(self.log_color, f"[{self.name}]", Style.RESET_ALL, *args, flush=flush, end=end)
    
    def stop(self):
        if self.current_env is not None:
            # self.log("Stopping current environment")
            self.current_env.stop()