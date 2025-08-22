
import random
from threading import Lock
from classes.EpisodeEnvironment import EpisodeEnvironment
from classes.Process import Process
from classes.Sample import Sample
from dataloading.EpisodeLoader import EpisodeLoader
import torch
from tqdm import tqdm
from queue import Queue
from transformers import DynamicCache


class TrainingProcess(Process):
    episode_loader: EpisodeLoader
    inference_queue: Queue
    sample_queue: Queue
    debug: bool
    lock: Lock
    mode: str
    
    def __init__(self, index: int, llm_service, transcription_service, episode_loader: EpisodeLoader, mode:str, debug: bool = False):
        super().__init__(index, llm_service, transcription_service)
        self.episode_loader = episode_loader
        self.inference_queue = Queue()
        self.sample_queue = Queue()
        self.debug = debug
        self.lock = Lock()
        self.mode = mode
        
    def load_new_episode_env(self, teacher_forcing: bool = False):
        """
        Load a new episode environment using the episode loader.
        """
        if self.current_env is not None:
            # self.log("Stopping old environment")
            self.current_env.stop()
        
        new_episode = self.episode_loader.next_episode()
        
        new_episode_env_id = f"P{self.index}-E{new_episode.call_tree_id}-G{new_episode.goal.correct_path_segment_ids[-1]}"
        # self.log("Loading new episode:", new_episode.call_tree_id, "with goal:", new_episode.goal.description, f"({new_episode_env_id})")
        
        self.current_env = EpisodeEnvironment(
            new_episode_env_id,
            new_episode,
            self.llm_service,
            self.transcription_service,
            self.inference_queue,
            self.sample_queue,
            log_color=self.log_color,
            mode=self.mode,
            teacher_forcing=teacher_forcing
        )
        self.current_env.start()
    
    def prepare_next_frame(self, teacher_forcing_p: float = 0.0):
        """
        Prepare the next frame for inference.
        """
        if self.current_env is None or self.current_env.done:
            teacher_forcing = random.random() < teacher_forcing_p
            self.load_new_episode_env(teacher_forcing)
            
        self.current_env.prepare_next_frame()
    
    def get_inference_input(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the input IDs and attention mask for the current environment.
        Returns:
            tuple: A tuple containing the input IDs, attention mask tensors and past_key_values.
        """
        input_ids = self.inference_queue.get()
        attention_mask = self.inference_queue.get()
        past_key_values = self.inference_queue.get()
        input_length = self.inference_queue.get()
        cache_position = self.inference_queue.get()
        
        return input_ids, attention_mask, past_key_values, input_length, cache_position

    def set_inference_output(self, process_output: torch.Tensor, value: float, past_key_values: DynamicCache, C:int):
        """
        Set the inference output for the current environment.
        """
        self.current_env.inference_output_queue.put(process_output)
        self.current_env.inference_output_queue.put(value)
        self.current_env.inference_output_queue.put(past_key_values)
        self.current_env.inference_output_queue.put(C)
        
    def collect_sample(self) -> Sample:
        """
        Collect a sample from the current environment.
        """
        sample = self.sample_queue.get()
        return sample