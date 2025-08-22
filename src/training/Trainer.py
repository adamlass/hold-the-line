import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from classes.Sample import Sample
from classes.TrainingProcess import TrainingProcess
from colorama import Fore, Style
from dataloading.EpisodeLoader import EpisodeLoader
from models.ModelIdentifier import ModelIdentifier
import numpy as np
from services.LLMService import LLMService
from services.TranscriptionService import TranscriptionService
import torch
from tqdm import tqdm
import wandb
from training.episode_storage import EpisodeStorage

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.99
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
# VALUE_LOSS_COEF = 1.0
CLIP_EPSILON = 0.2
MAX_GRAD_NORM = 0.5
ADAM_EPSILON = 1e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

LEARNING_RATE = 5e-7
LEARNING_RATE_VALUE = 1e-5

class Trainer:
    llm_service: LLMService
    transcription_service: TranscriptionService
    episode_loader: EpisodeLoader
    training_processes: list[TrainingProcess]
    model_name: str
    lora_enabled: bool
    quantized: bool
    lora_r: int
    tag: str
    learning_rate: float
    learning_rate_value: float
    dropout: float
    entropy_coef: float
    save: bool
    normalize_advantages: bool
    mode: str
    teacher_forcing: bool
    teacher_forcing_steps: int
    teacher_forcing_p: float
    unfreezed_layers: int
    
    def __init__(self, model_name: str, lora_enabled: bool = True, quantized: bool = False, lora_r:int = 16, lora_alpha: int = 32, tag: str = None, dropout:float = 0.0, value_dropout: float = 0.0, entropy_coef: float = ENTROPY_COEF, save: bool = False, learning_rate: float = LEARNING_RATE, learning_rate_value: float = LEARNING_RATE_VALUE, normalize_advantages: bool = False, mode: str = "full", teacher_forcing_steps: int = 0, unfreezed_layers:int = 0, save_samples: bool = False, log_color: str = Fore.YELLOW):
        self.llm_service = LLMService(model_name, lora_enabled, quantized=quantized, lora_r=lora_r, lora_alpha=lora_alpha, dropout=dropout, value_dropout = value_dropout, unfreezed_layers=unfreezed_layers, mode=mode)
        self.transcription_service = None
        self.episode_loader = EpisodeLoader()
        self.training_processes = []
        self.log_color = log_color
        self.model_name = model_name
        self.lora_enabled = lora_enabled
        self.quantized = quantized
        self.lora_r = lora_r
        self.tag = tag
        self.dropout = dropout
        self.lora_alpha = lora_alpha
        self.value_dropout = value_dropout
        self.learning_rate = learning_rate
        self.learning_rate_value = learning_rate_value
        self.entropy_coef = entropy_coef
        self.save = save
        self.normalize_advantages = normalize_advantages
        self.mode = mode
        self.teacher_forcing = teacher_forcing_steps > 0
        self.teacher_forcing_steps = teacher_forcing_steps
        self.teacher_forcing_p = 0.0
        self.unfreezed_layers = unfreezed_layers
        self.save_samples = save_samples
        self.episode_storage = None
        
        # Create training folder and episode storage if saving samples
        if self.save_samples:
            self.training_dir = self._create_training_folder()
            # Initialize episode storage with configuration
            config = {
                'model': model_name,
                'lora_enabled': lora_enabled,
                'quantized': quantized,
                'learning_rate': learning_rate,
                'learning_rate_value': learning_rate_value,
                'mode': mode,
                'teacher_forcing_steps': teacher_forcing_steps
            }
            self.episode_storage = EpisodeStorage(self.training_dir, tag, config)
        else:
            self.training_dir = None
    
    def _create_training_folder(self):
        """Create a new training folder with timestamp and tag."""
        # Create base data directory if it doesn't exist
        base_dir = Path("data/trainings")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training-specific folder with timestamp and tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"training_{self.tag if self.tag else 'unnamed'}_{timestamp}"
        training_dir = base_dir / folder_name
        training_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Created training folder: {training_dir}")
        return training_dir
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(self.log_color, *args, Style.RESET_ALL, flush=flush, end=end)
        
    def initialize_training_processes(self, process_count: int):
        for index in range(process_count):
            process = TrainingProcess(index, 
                                      self.llm_service, 
                                      self.transcription_service, 
                                      self.episode_loader,
                                      mode=self.mode)
            self.training_processes.append(process)
    
    def new_or_resumed_training(self, model_identifier: ModelIdentifier = None, config: dict = None):
        wandb.init(project=WANDB_PROJECT_NAME,
                    entity=WANDB_ENTITY,
                    resume="must" if model_identifier is not None else "allow",
                    id=model_identifier.run_id if model_identifier is not None else None,
                    config=config,
                    )
        run_id = wandb.run.name
        
        if model_identifier is None:
            self.log(f"Starting new training run {run_id}")
        else:
            self.log(f"Resuming training from run {run_id}")
            
        wandb.define_metric("sample_step")
        sb = self.dictify_my_batch([])
        for key in sb:
            wandb.define_metric(key, step_metric="sample_step")
    
    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics)
        
    def add_tag(self, tag: str):
        wandb.run.tags = wandb.run.tags + (tag,)
    
    def run_training(self, total_updates: int, mini_batch_frames: int, mini_batch_epochs: int, batch_size: int, resume_model: ModelIdentifier = None):
        last_mini_batch = []
        total_frames_sampled = 0
        total_mini_batch_epochs = 0
        self.optimizer = torch.optim.AdamW([
            {"params": self.llm_service.model.base_model.parameters(), "lr": self.learning_rate},
            {"params": self.llm_service.model.value_head.parameters(), "lr": self.learning_rate_value},
        ],
            eps=ADAM_EPSILON,
            betas=(ADAM_BETA1, ADAM_BETA2)
        )
        
        self.new_or_resumed_training(resume_model, config={
            "model_name": self.model_name,
            "lora_enabled": self.lora_enabled,
            "quantized": self.quantized,
            "total_updates": total_updates,
            "mini_batch_frames": mini_batch_frames,
            "mini_batch_epochs": mini_batch_epochs,
            "batch_size": batch_size,
            "tag": self.tag,
            "learning_rate": self.learning_rate,
            "learning_rate_value": self.learning_rate_value,
            "lora_r": self.lora_r,
            "dropout": self.dropout,
            "entropy_coef": self.entropy_coef,
            "lora_alpha": self.lora_alpha,
            "value_dropout": self.value_dropout,
            "save": self.save,
            "normalize_advantages": self.normalize_advantages,
            "observation_mode": self.mode,
            "teacher_forcing": self.teacher_forcing,
            "teacher_forcing_steps": self.teacher_forcing_steps,
            "unfreezed_layers": self.unfreezed_layers,
        })
        
        if self.lora_enabled:
            self.add_tag("lora")
            self.add_tag(f"lora_r_{self.lora_r}")
        
        if self.quantized: self.add_tag("quantized")
            
        self.add_tag(self.llm_service.device)

        best_loss = float("inf")
        try:
            for update_i in tqdm(range(total_updates), desc="Training Updates", leave=False):
                samples_required = mini_batch_frames + 1
                if self.teacher_forcing:
                    self.teacher_forcing_p = max(0.0, 1.0 - (update_i / self.teacher_forcing_steps))
                mini_batch = self.sample_mini_batch(samples_required)
                extra_frame = mini_batch.pop(-1)
                
                # Save extra frame samples to episode storage
                if self.episode_storage:
                    for process_i, sample in enumerate(extra_frame):
                        sample_dict = sample.to_dict()
                        sample_dict['process_id'] = process_i
                        self.episode_storage.add_sample(process_i, sample_dict, is_extra_frame=True)
                
                last_mini_batch = mini_batch
                self.set_advantages(mini_batch, extra_frame)
                self.print_mini_batch(mini_batch)
                
                flattened_mini_batch = np.array(mini_batch).flatten()
                total_frames_sampled += len(flattened_mini_batch)
                
                # Increment sample step in episode storage
                if self.episode_storage:
                    self.episode_storage.increment_sample_step()
                
                self.log_mini_batch_stats(flattened_mini_batch, step=update_i)
                
                for mini_batch_epoch_i in tqdm(range(mini_batch_epochs), desc="Minibatch Epochs", leave=False):
                    batches_starting_indexes = self.get_batches_starting_indexes(len(flattened_mini_batch), batch_size)
                    
                    dict_returns = {}
                    for batch_starting_indexes in tqdm(batches_starting_indexes, desc=f"Updating with Batches of size {batch_size}", leave=False):
                        current_batch = flattened_mini_batch[batch_starting_indexes]
                        dict_return = self.update_model(current_batch)
                        for key in dict_return:
                            if key not in dict_returns:
                                dict_returns[key] = []
                            dict_returns[key].append(dict_return[key])
                        
                    for key in dict_returns:
                        dict_returns[key] = np.mean(dict_returns[key])
            
                    current_loss = dict_returns["loss"]
                    if self.save and current_loss < best_loss:
                        self.log(f"Saving model with new best loss: {current_loss}")
                        self.llm_service.save_model(self.tag if self.tag is not None else "best")
                        best_loss = current_loss
                        
                    dict_returns["frames"] = total_frames_sampled
                    dict_returns["update"] = update_i
                    dict_returns["mini_batch_epoch"] = total_mini_batch_epochs
                    self.log_metrics(dict_returns)
                    total_mini_batch_epochs += 1
                
        except KeyboardInterrupt:
            self.log("Keyboard interrupt!")
        except Exception as e:
            self.log(f"An error occurred: {e}")
            self.log("LAST MINIBATCH SAMPLES:")
            self.print_mini_batch(last_mini_batch)
        finally:
            # Finalize episode storage
            if self.episode_storage:
                self.episode_storage.finalize()
                
            for process in self.training_processes:
                self.log(f"Stopping process {process.name}")
                process.stop()
                self.log(f"Process {process.name} stopped")
            self.log("Stopping wandb run")
            wandb.finish()
            self.log("Wandb run stopped")
    
    def print_mini_batch(self, mini_batch: list[list[Sample]]):
        for frame_samples in mini_batch:
            for sample in frame_samples:
                self.log(sample)
            
    def log_mini_batch_stats(self, mini_batch: list[list[Sample]], step=None):
        sb = self.dictify_my_batch(mini_batch)
        sb["done_reward"] = sb["reward"][sb["next_done"]]
        done_accuracy = sb["done_reward"] == 1.0
        sb["done_accuracy"] = done_accuracy.float()
        sb["teacher_forced"] = sb["teacher_forced"].float()
        del sb["next_done"]
        stds = {}
        for key, value in sb.items():
            if key == "action" or key == "prompt_length":
                length = value.shape[0]
                sum = torch.sum(value).item()
                sb[key] = sum / length
                continue
            sb[key] = torch.mean(value, dim=-1).item()
            stds[f"{key}_std"] = torch.std(value, dim=-1).item()
        sb.update(stds)
        sb["sample_step"] = step
        sb["teacher_forcing_p"] = self.teacher_forcing_p
        self.log_metrics(sb)
     
    def update_model(self, batch: list[Sample]):
        input_lens = [int(sample.prompt_ids.shape[1]) for sample in batch]
        samples_input_ids = [sample.prompt_ids for sample in batch]
        
        batch_input_ids = []
        batch_attention_masks = []
        for sample_input_ids in samples_input_ids:
            sample_attention_mask = torch.ones_like(sample_input_ids)
            input_ids, attention_mask  = self.llm_service.get_frame_batch(sample_input_ids, sample_attention_mask)
            batch_input_ids.extend(input_ids)
            batch_attention_masks.extend(attention_mask)
        
        combined_input_ids, combined_attention_masks = self.llm_service.combine_inputs(batch_input_ids, batch_attention_masks)
        
        output_logits, values = self.llm_service.batched_inference(combined_input_ids,
                                           combined_attention_masks,
                                           input_lens,
                                           training=True)
        
        input_splits, output_splits = self.llm_service.split_io(combined_input_ids, output_logits)
        action_logits_list = [self.llm_service.get_action_logits(input_ids, output_split, input_len) for input_ids, output_split, input_len in zip(input_splits, output_splits, input_lens)]
        action_logits = torch.stack(action_logits_list, dim=0).squeeze()
        distribution = self.llm_service.get_distribution(action_logits)
        
        sb = self.dictify_my_batch(batch)
        
        # value loss
        value_clipped = sb['value'] + torch.clamp(values - sb['value'], -CLIP_EPSILON, CLIP_EPSILON)
        surr_v1 = (values - sb['returnn']).pow(2)
        surr_v2 = (value_clipped - sb['returnn']).pow(2)
        value_loss = torch.max(surr_v1, surr_v2).mean()
        
        # policy loss
        entropy = distribution.entropy()
        entropy_loss = entropy.mean()
        log_prob = distribution.log_prob(sb["action"])
        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)
            
        with torch.no_grad():
            wait_log_prob = distribution.log_prob(torch.zeros_like(sb["action"])).mean()
            press_1_log_prob = distribution.log_prob(torch.ones_like(sb["action"])).mean()
            press_2_log_prob = distribution.log_prob(torch.ones_like(sb["action"]) * 2).mean()
            approx_kl = (sb["log_prob"] - log_prob).mean()
        
        advantages = sb['advantage']
        
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ratio = torch.exp(log_prob - sb['log_prob'])
        surr1 = ratio * advantages
        
        clipped_ratio = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
        surr2 = clipped_ratio * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()
        loss: torch.Tensor = policy_loss - self.entropy_coef * entropy_loss + VALUE_LOSS_COEF * value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(
            p.grad.data.detach().cpu().norm(2) ** 2 for p in self.llm_service.model.parameters() if
            p.grad is not None) ** MAX_GRAD_NORM
        # clip_grad_norm = torch.nn.utils.clip_grad_norm_(self.llm_service.model.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        dict_return = {"loss": loss.item(),
                       "learning_rate": self.optimizer.param_groups[0]['lr'],
                       "learning_rate_value": self.optimizer.param_groups[1]['lr'],
                        "entropy": entropy_loss.item(),
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "grad_norm": grad_norm.item(),
                        # "clip_grad_norm": clip_grad_norm.item(),
                        "wait_log_prob": wait_log_prob.item(),
                        "press_1_log_prob": press_1_log_prob.item(),
                        "press_2_log_prob": press_2_log_prob.item(),
                        "approx_kl": approx_kl.item(),
                        "ratio":ratio.mean().item(),
                        "ratio_std": ratio.std().item(),}
        return dict_return
            
    def dictify_my_batch(self, batch: list[Sample]) -> dict:
        sb = {
            "reward": [],
            "action": [],
            "value": [],
            "log_prob": [],
            "advantage": [],
            "returnn": [],
            "prompt_length": [],
            "next_done": [],
            "teacher_forced": [],
        }
        
        for sample in batch:
            sb["reward"].append(sample.reward)
            sb["action"].append(sample.action.index)
            sb["value"].append(sample.value)
            sb["log_prob"].append(sample.log_prob)
            sb["advantage"].append(sample.advantage)
            sb["returnn"].append(sample.returnn)
            sb["prompt_length"].append(sample.prompt_ids.shape[1])
            sb["next_done"].append(sample.next_done)
            sb["teacher_forced"].append(sample.teacher_forced)
        
        for key in sb:
            sb[key] = torch.tensor(sb[key], device=self.llm_service.device)
         
        return sb
    
    def get_batches_starting_indexes(self, total_samples: int, batch_size: int) -> list[int]:
        indexes = np.arange(0, total_samples)
        random_indexes = np.random.permutation(indexes)
        batches_starting_indexes = [random_indexes[i:i + batch_size] for i in range(0, len(random_indexes), batch_size)]

        return batches_starting_indexes
    
    def set_advantages(self, mini_batch: list[list[Sample]], extra_frame: list[Sample]) -> list[list[Sample]]:
        next_values = {}
        next_advantages = {}
        
        mini_batch_length = len(mini_batch)
        
        for process_i, sample in enumerate(extra_frame):
            next_values[process_i] = sample.value
            next_advantages[process_i] = 0
        
        for frame_i in reversed(range(mini_batch_length)):
            frame_samples: list[Sample] = mini_batch[frame_i]
            for process_i, sample in enumerate(frame_samples):
                next_mask = 0 if sample.next_done else 1
                next_value = next_values[process_i]
                next_advantage = next_advantages[process_i]
                
                reward = sample.reward
                value = sample.value
                
                delta = reward + DISCOUNT_FACTOR * next_value * next_mask - value
                advantage = delta + DISCOUNT_FACTOR * GAE_LAMBDA * next_advantage * next_mask
                
                sample.advantage = advantage
                sample.returnn = value + advantage
                sample.process_id = process_i
                
                next_values[process_i] = value
                next_advantages[process_i] = advantage
                
    def sample_mini_batch(self, mini_batch_frames: int) -> list[list[Sample]]:
        mini_batch: list[list[Sample]] = []
        
        for frame_i in tqdm(range(mini_batch_frames), desc="Collecting Frames", leave=False):
            # For the last frame (which will be used as extra frame), don't save to storage yet
            is_last_frame = (frame_i == mini_batch_frames - 1)
            frame_samples: list[Sample] = self.sample_frame(save_to_storage=not is_last_frame)
            mini_batch.append(frame_samples)
        
        return mini_batch
    
    def sample_frame(self, save_to_storage: bool = True) -> list[Sample]:
        frame_samples = []
        
        for process in tqdm(self.training_processes, desc="Collecting Frame in Processes", leave=False):
            process.prepare_next_frame(self.teacher_forcing_p)
            sample: Sample = process.collect_sample()
            frame_samples.append(sample)
            
            # Save sample to episode storage only if requested
            if self.episode_storage and save_to_storage:
                sample_dict = sample.to_dict()
                sample_dict['process_id'] = process.index
                self.episode_storage.add_sample(process.index, sample_dict)
            
        return frame_samples
    
    # Removed old save_samples_to_json method - now using EpisodeStorage
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=32) # Number of training processes - default 32 as in the paper
    parser.add_argument("--frames", type=int, default=40) # Number of frames in the mini batch - default 40 as in the paper
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4) # Mini batch epochs - default 4 as in the paper
    parser.add_argument("--batch_size", type=int, default=8) # TODO should be 64 as in the paper, but gives OOM right now
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--dropout", type=int, default=0)
    parser.add_argument("--vdropout", type=int, default=0)
    parser.add_argument("--entropy", type=int, default=1)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--lora", action="store_true", help="Enable LoRA finetuning")
    parser.add_argument("--quantized", action="store_true", help="Enable quantization")
    parser.add_argument("--save", action="store_true", help="Save best model during training")
    parser.add_argument("--save_samples", action="store_true", help="Save training samples to JSON file for demo replay")
    parser.add_argument("--norm_adv", action="store_true", help="Normalize advantages during training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate for the policy optimizer")
    parser.add_argument("--lr_value", type=float, default=LEARNING_RATE_VALUE, help="Learning rate for the value head (critic) optimizer")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "partial", "complete"], help="Observation mode under training: full, partial or complete")
    parser.add_argument("--tf_steps", type=int, default=0, help="Number of teacher forcing steps during training (Linear decay)")
    parser.add_argument("--unfreezed_layers", type=int, default=0, help="Number of unfreezed layers in the model (1 means that last layer is unfreezed)")
    
    args = parser.parse_args()
    print("args", args)
    N_TRAINING_PROCESSES = args.processes
    MINI_BATCH_FRAMES = args.frames
    TOTAL_UPDATES = args.updates
    MINI_BATCH_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    MODEL_NAME = args.model
    LORA = args.lora
    QUANTIZED = args.quantized
    TAG = args.tag
    SAVE = args.save
    SAVE_SAMPLES = args.save_samples
    NORMALIZE_ADVANTAGES = args.norm_adv
    LEARNING_RATE = args.lr
    LEARNING_RATE_VALUE = args.lr_value
    MODE = args.mode
    TEACHER_FORCING_STEPS = args.tf_steps
    UNFREEZED_LAYERS = args.unfreezed_layers
    
    entropy_coef: float = args.entropy / 100.0
    dropout: float = args.dropout / 100.0
    value_dropout: float = args.vdropout / 100.0

    trainer = Trainer(model_name=MODEL_NAME, lora_enabled=LORA, quantized=QUANTIZED, lora_r=LORA_R, lora_alpha=LORA_ALPHA, tag=TAG, dropout=dropout, value_dropout=value_dropout, entropy_coef=entropy_coef, save=SAVE, learning_rate=LEARNING_RATE, learning_rate_value=LEARNING_RATE_VALUE, normalize_advantages=NORMALIZE_ADVANTAGES, mode=MODE, teacher_forcing_steps= TEACHER_FORCING_STEPS, unfreezed_layers=UNFREEZED_LAYERS, save_samples=SAVE_SAMPLES)
    trainer.initialize_training_processes(N_TRAINING_PROCESSES)
    trainer.run_training(TOTAL_UPDATES, MINI_BATCH_FRAMES, MINI_BATCH_EPOCHS, BATCH_SIZE)
