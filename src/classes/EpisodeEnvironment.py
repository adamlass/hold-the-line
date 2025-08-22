
from typing import Iterator
from classes.Action import Action
from classes.Dial import Action
from classes.Environment import Environment
from classes.Episode import Episode
from classes.EpisodeStateMachine import EpisodeStateMachine
from classes.ToyEpisodeStateMachine import ToyEpisodeStateMachine
from classes.Sample import Sample
from colorama import Fore
import numpy as np
from services.LLMService import LLMService
from services.TranscriptionService import TranscriptionService
import torch
from transceivers.CallClient import CallClient
from transceivers.LLMAgent import LLMAgent
from transceivers.MotorFunction import MotorFunction
from transceivers.Transcriber import Transcriber
from queue import Queue
from transformers import DynamicCache
from threading import Lock
from torch.distributions import Categorical


CHUNK_SIZE = 1024         # bytes per chunk (as in IVRService)
SAMPLE_RATE = 16000       # Hz
CHANNELS = 1              # mono
SAMPLE_WIDTH = 4          # bytes per sample
D_TYPE = np.float32

class EpisodeEnvironment(Environment):
    episode_iterator: Iterator[Episode]
    call_client: CallClient
    transcriber: Transcriber
    llm_agent: LLMAgent
    motor_fn: MotorFunction
    episode: Episode
    call_sm: EpisodeStateMachine
    current_sample: Sample
    sample_lock: Lock
    inference_queue: Queue
    inference_output_queue: Queue
    sample_queue: Queue
    done: bool
    teacher_forcing: bool

    def __init__(self, id: str, episode: Episode, llm_service: LLMService, transcription_service: TranscriptionService, inference_queue: Queue, sample_queue:Queue, mode: str, teacher_forcing: bool = False, log_color: str = Fore.MAGENTA):
        super().__init__(id, llm_service, transcription_service, log_color=log_color)
        self.episode = episode
        self.sample_lock = Lock()
        self.current_sample = None
        self.inference_queue = inference_queue
        self.inference_output_queue = Queue()
        self.sample_queue = sample_queue
        self.done = False
        self.call_sm = EpisodeStateMachine(self.episode, mode)
        self.teacher_forcing = teacher_forcing
        
    def start(self):
        # self.call_client = CallClient(self)
        # self.transcriber = Transcriber(self)
        self.llm_agent = LLMAgent(self)
        self.motor_fn = MotorFunction(self)
        
        # self.call_client.subscribe(self.llm_agent)
        # self.transcriber.subscribe(self.llm_agent)
        self.llm_agent.subscribe(self.motor_fn)
    
        # call = Call(self.id, self.episode.call_tree_id, self.call_client.input_queue)
        
        self.motor_fn.run()
        self.llm_agent.run()
        # self.transcriber.run()
        # self.call_client.run()
        
    def stop(self):
        # self.call_client.join()
        # self.transcriber.join()
        self.llm_agent.join()
        self.motor_fn.join()
    
    def apply_action(self, action: Action):
        with self.sample_lock:
            reward, done = self.call_sm.apply_action(action)
            # self.log("Action:", action.text, "Reward:", reward, "Done:", done)
            self.current_sample.reward = reward
            self.current_sample.next_done = done
            self.done = done
            
        self.sample_queue.put(self.current_sample)
        
    def get_user_goal(self) -> str:
        return self.episode.goal.description
    
    def prompt(self, input_ids: torch.Tensor, input_attention_mask: torch.Tensor, past_key_values: DynamicCache, cache_position: int) -> tuple[Action, DynamicCache, int]:
        with self.sample_lock:
            input_length = input_ids.shape[1]
            batch_input_ids, batch_attention_mask = self.llm_service.get_frame_batch(input_ids, input_attention_mask)
            
            cropped_input_ids, cropped_past_key_values, C = self.llm_service.crop_input_and_cache(batch_input_ids, past_key_values, cache_position)
            cropped_input_len = input_length - C
            
            output_logits, values = self.llm_service.batched_inference(
                cropped_input_ids,
                batch_attention_mask,
                input_lengths=[cropped_input_len],
                past_key_values=cropped_past_key_values,
                training=False
                )
            
            value = values.item()
            
            action_logits = self.llm_service.get_action_logits(cropped_input_ids, output_logits, cropped_input_len)
            distribution = self.llm_service.get_distribution(action_logits)
            
            if self.teacher_forcing:
                action_index = self.call_sm.get_best_action_index()
                action_index = torch.tensor(action_index, device=self.llm_service.device, dtype=torch.int64)
                log_prob = distribution.log_prob(action_index).item()
                action = self.llm_service.actions[action_index]
            else:
                action_index, log_prob = self.llm_service.sample_action_index_from_distribution(distribution)
                action = self.llm_service.actions[action_index]
            
            self.current_sample = Sample(
                action=action,
                value=value,
                reward=None,
                prompt_ids=input_ids.cpu(),
                next_done=None,
                log_prob=log_prob,
                advantage=None,
                process_id=None,
                returnn=None,
                teacher_forced=self.teacher_forcing,
                observation_text=getattr(self, 'current_observation_text', None),
                user_goal=self.episode.goal.description,
                call_tree_id=self.episode.call_tree_id
                )
            
            new_cache_position = input_length - 1
            return action, cropped_past_key_values, new_cache_position
    
    def prepare_next_frame(self):
        with self.sample_lock:
            self.current_sample = None
            text: str = self.call_sm.next_frame()
            self.current_observation_text = text  # Store for later use in sample
            self.llm_agent.input_queue.put(text)
            