from abc import ABC

from classes.Action import Action
from classes.ActionTokenResultTree import ActionTokenResultTree
from classes.ActionTokenTree import ActionTokenTree
from classes.ActionType import ActionType
from colorama import Fore, Style
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils import get_best_device
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, BitsAndBytesConfig
from threading import Lock
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODELS_DIR = "data/models"

SYSTEM_PROMPT = """Available actions of the assistant: {}
Assistant goal: Help the human navigate the IVR phone menu, based on their goal.
Human goal: '{}'

If you are unsure about the most appropriate action, wait for input.

The following is speech-to-text transcription chunks from the phone call:
"""

ACTIONS: list[str] = [str(action.value) for action in ActionType]
DEFAULT_NUM_DIAL_ACTIONS = 9
            
class LLMService(ABC):
    inference_lock = Lock()
    actions: list[Action]
    actions_token_ids: list[list[int]]
    action_tree: ActionTokenTree
    action_tree_depth: int
    action_input_ids_appendix: torch.Tensor
    action_attention_mask_appendix: torch.Tensor
    lora_enabled: bool
    newline_token_id: int
    dropout: float
    lora_alpha: int
    value_dropout: float
    mode: str

    def __init__(self, model_name: str, lora_enabled: bool = True, quantized: bool = False, lora_r: int = 16, lora_alpha:int = 32, dropout: float = 0.0, value_dropout: float = 0.0, num_dial_actions: int = DEFAULT_NUM_DIAL_ACTIONS, unfreezed_layers: int = 0, mode: str = "full"):
        self.name = f"LLMService"
        self.log_color = Fore.BLUE
        self.lora_enabled = lora_enabled
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        self.value_dropout = value_dropout
        self.mode = mode
        self.actions_str = ", ".join(ACTIONS)
        self.log(f"Available Actions: {self.actions_str}")
        
        self.device = get_best_device()
        self.log("Using device:", self.device)
        
        self.log(f"Setting up LLMService using model: {model_name}")
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        quantization_config = None
        device_map = None if self.device == "mps" else "auto" # auto doesn't work on MPS
        torch_dtype = None if self.device == "mps" else torch.float16 # float16 doesn't work on MPS
        
        if quantized:
            self.log("Using quantized model")
            quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            device_map = "auto"
            torch_dtype = torch.bfloat16
            
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              device_map=device_map,
                                                              quantization_config=quantization_config,
                                                              torch_dtype=torch_dtype)
        if lora_enabled:
            self.log("Using LoRA")
            self.log("Lora r:", lora_r)
            self.log("Freezing LLM model parameters")
            
            for param in self.llm_model.parameters():
                param.requires_grad = False
                

            lora_config = LoraConfig(
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules = ["q_proj", "v_proj"],
                lora_dropout = self.dropout, # 0.05 originally
                bias = "none",
                task_type = "CAUSAL_LM"
            )
            self.llm_model.gradient_checkpointing_enable()
            self.llm_model.enable_input_require_grads()
            self.log("Adding LoRA adapters")
            self.model = get_peft_model(self.llm_model, lora_config)
            self.model.config.use_cache = False
        else:
            self.log("Using LLM model without LoRA adapters")
            self.model = self.llm_model
        
        self.print_trainable_parameters(self.model)
        
        # unfreezing the last `unfreezed_layers` layers
        for name, param in self.model.named_parameters():
            if "model.layers" in name:
                layer_number = int(name.split("layers")[1].split(".")[1])
                if layer_number >= self.llm_model.config.num_hidden_layers - unfreezed_layers:
                    param.requires_grad = True
                    self.log(f"Unfreezing layer {layer_number} ({name})")
        
        self.print_trainable_parameters(self.model)
        
        self.log("Adding Value Head")
        llm_hidden_size = self.model.config.hidden_size
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.value_dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.value_dropout),
            torch.nn.Linear(512, 1))
        # self.value_head_op = nn.Sequential(
        #     nn.Linear(llm_hidden_size, 2048),  # wider first layer
        #     nn.GELU(),
        #     nn.LayerNorm(2048),                # stabilises activations
        #     nn.Dropout(self.value_dropout),    # keep 0.0-0.1
        #     nn.Linear(2048, 1)              # final output layer
        # )
        self.model.value_head = self.value_head_op
        
        self.print_trainable_parameters(self.model)
        
        self.log("Sending model to device:", self.device)
        self.model.to(self.device)
        self.log("Model sent to device:", self.device)
        
        self.log("Generating actions")
        self.actions = self.generate_actions(num_dial_actions=num_dial_actions)
        
        self.log("Generating action tree")
        self.actions_token_ids = [action.token_ids.tolist() for action in self.actions]
        self.action_tree_depth = max([len(action.token_ids) for action in self.actions])
        self.action_tree = self.generate_action_tree(None, self.actions_token_ids)
        
        self.log("Generating action tree batch appendix")
        action_input_ids_appendix, action_attention_mask_appendix = self.create_appendix(self.actions_token_ids)
        self.action_input_ids_appendix = action_input_ids_appendix.to(torch.int64)
        self.action_attention_mask_appendix = action_attention_mask_appendix.to(torch.int64)
        self.process_batch_size = self.action_input_ids_appendix.shape[0]
        self.log("Batch size:", self.process_batch_size)
        
        self.env_past_key_values = {}
        self.env_cache_positions = {}
        
        self.newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
        self.log("Newline token ID:", self.newline_token_id)
    
    def log(self, *args, flush: bool=True, end: str="\n"):
        print(f"{self.log_color}{self.name}:", *args, Style.RESET_ALL, flush=flush, end=end)
    
    def save_model(self, model_name: str):
        # self.log("Saving model to", model_name)
        model_to_save = self.model
        if self.lora_enabled:
            model_to_save = self.model.merge_and_unload()
        model_to_save.save_pretrained(f"{MODELS_DIR}/{model_name}")
    
    def get_system_prompt(self, goal: str) -> str:
        """
        Get the system prompt for the LLM.
        """
        system_prompt = SYSTEM_PROMPT.format(self.actions_str, goal)
        return system_prompt
    
    def apply_chat_template(self, chat: list[dict[str, str]], add_generation_prompt: bool = True) -> torch.Tensor:
        # self.log("Applying chat template - acquiring inference lock")
        with self.inference_lock:
            # self.log("Applying chat template")
            # if self.mode == "partial":
            #     inputs = self.tokenizer(chat[0]["content"], add_special_tokens=False, return_tensors="pt", return_token_type_ids=False, return_attention_mask=True)
            # else:
            inputs = self.tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, add_generation_prompt=add_generation_prompt, return_tensors="pt", add_special_tokens=True)
            
            # self.log("Sending inputs to device", self.device)
            
            inputs = inputs.to(self.device)
            # self.log("Inputs sent to device", self.device)
            return inputs
    
    def batch_decode(self, input_ids: torch.Tensor) -> list[str]:
        with self.inference_lock:
            decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            return decoded
        
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        self.log(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    def generate_actions(self, num_dial_actions: int) -> list[Action]:
        action_strings: list[str] = []
        for action_str in ACTIONS:
            if "[number]" in action_str:
                for i in range(1, num_dial_actions + 1):
                    number = i % 10
                    action_copy = str(action_str)
                    action_copy = action_copy.replace("[number]", str(number), 1)
                    action_strings.append(action_copy)
            else:  
                action_strings.append(action_str)
        
        actions: list[Action] = []
        for index, action_str in enumerate(action_strings):
            token_ids = self.tokenizer(action_str, return_tensors="pt", add_special_tokens=False)
            token_ids = token_ids["input_ids"].squeeze(0).cpu()
            action = Action(index, action_str, token_ids)
            actions.append(action)
            
        return actions
    
    def generate_action_tree(self, token_id: int, token_ids_list: list[list[int]]) -> ActionTokenTree:
        token_mapping = {}
        for token_ids in token_ids_list:
            if len(token_ids) == 0:
                continue
            
            first_token_id = token_ids[0]
            if first_token_id not in token_mapping:
                token_mapping[first_token_id] = []
                
            token_mapping[first_token_id].append(token_ids[1:])
        
        children: list[ActionTokenTree] = []
        for child_token_id, children_token_ids in token_mapping.items():
            child = self.generate_action_tree(child_token_id, children_token_ids)
            children.append(child)
        return ActionTokenTree(token_id, children)
    
    def set_probabilities_in_tree(self, tree: ActionTokenTree, logit_token_ids, all_logits, probability: float = None) -> ActionTokenResultTree:
        result = ActionTokenResultTree(tree.token_id, [], probability)
        
        if len(all_logits) == 0:
            return result
        
        cur_logit = all_logits[0, 0]
        children_token_ids = logit_token_ids[:, 1:]
        children_logits = all_logits[:, 1:]
        
        for child in tree.children:
            child_token_id = child.token_id
            child_probability = cur_logit[child_token_id].item()
            
            child_indexes = [i for i in range(len(children_token_ids)) if len(children_token_ids[i]) > 0 and children_token_ids[i][0] == child_token_id]
            child_token_ids = children_token_ids[child_indexes]
            child_logits = children_logits[child_indexes]
            
            child_result = self.set_probabilities_in_tree(child, child_token_ids, child_logits, child_probability)
            result.children.append(child_result)
            
        return result
    
    def create_appendix(self, actions_token_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        actions_token_ids_t = [torch.Tensor(action_token_ids[: -1]).T for action_token_ids in actions_token_ids if len(action_token_ids) > 1]
        input_ids_appendix = pad_sequence(actions_token_ids_t, batch_first=True, padding_value=0)
        input_ids_appendix = torch.unique(input_ids_appendix, dim=0)
        input_ids_appendix = input_ids_appendix.to(device=self.device, dtype=torch.int64)
        attention_mask_appendix = torch.sign(input_ids_appendix.abs())
        return input_ids_appendix, attention_mask_appendix
        
    def get_action_token_probs(self, tree: ActionTokenResultTree, action_token_ids: list[int]) -> list[float]:
        result = []
        
        if tree.token_id is not None:
            result.append(tree.probability)
        
        if(len(action_token_ids) == 0):
            return result
        
        for child in tree.children:
            if child.token_id == action_token_ids[0]:
                child_results = self.get_action_token_probs(child, action_token_ids[1:])
                result.extend(child_results)
                break
            
        return result
    
    def get_frame_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        tree_input_ids = input_ids.repeat(self.process_batch_size, 1)
        tree_input_ids = torch.cat([tree_input_ids, self.action_input_ids_appendix], dim=1)
        
        tree_attention_mask = attention_mask.repeat(self.process_batch_size, 1)
        tree_attention_mask = torch.cat([tree_attention_mask, self.action_attention_mask_appendix], dim=1)
        
        return tree_input_ids, tree_attention_mask
    
    def combine_inputs(self, input_ids_list: list[torch.Tensor], attention_mask_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids_list_t = [input_ids.T for input_ids in input_ids_list]
        combined_input_ids = pad_sequence(input_ids_list_t, batch_first=True, padding_value=0)
        combined_input_ids = combined_input_ids.squeeze(-1).to(self.device)
        
        attention_mask_list_t = [attention_mask.T for attention_mask in attention_mask_list]
        combined_attention_masks = pad_sequence(attention_mask_list_t, batch_first=True, padding_value=0)
        combined_attention_masks = combined_attention_masks.squeeze(-1).to(self.device)
        
        return combined_input_ids, combined_attention_masks
    
    def split_io(self, input_ids: torch.Tensor, output_logits: torch.Tensor) -> list[torch.Tensor, torch.Tensor]:
        input_splits = input_ids.split(self.process_batch_size)
        output_splits = output_logits.split(self.process_batch_size)
        return input_splits, output_splits
    
    def get_distribution(self, action_sum_logits: torch.Tensor) -> Categorical:
        distribution = Categorical(logits=action_sum_logits)
        return distribution
    
    def get_action_logits(self, input_ids: torch.Tensor, output: torch.Tensor, input_length: int) -> torch.Tensor:
        last_prompt_token_index = input_length - 1
        action_token_ids = input_ids[:, last_prompt_token_index: last_prompt_token_index + self.action_tree_depth]
        action_token_logits = output[:, last_prompt_token_index: last_prompt_token_index + self.action_tree_depth, :]
        
        result_tree = self.set_probabilities_in_tree(self.action_tree, action_token_ids, action_token_logits)
        
        action_token_log_sums = []
        
        for action in self.actions:
            action_token_probs = self.get_action_token_probs(result_tree, action.token_ids)
            # self.log(f"Action token probabilities for {action.text}: {action_token_probs}")
            # action_token_logs = [np.log(prob) for prob in action_token_probs]
            action_token_log_sums.append(np.sum(action_token_probs))
        
        # special softmax - see GLAM paper for details
        # log_prob_sums = [np.exp(action_token_log_sum) for action_token_log_sum in action_token_log_sums]
        # sum_log_prob_sums = np.sum(log_prob_sums)
        # softmaxed_logits = log_prob_sums / sum_log_prob_sums
        action_sum_logits = torch.tensor(action_token_log_sums, dtype=torch.float32, device=self.device)
        
        return action_sum_logits
    
    def sample_action_index_from_distribution(self, distribution: Categorical) -> tuple[Action, float, float]:
        action_sample = distribution.sample()
        action_index = action_sample.item()
        
        log_prob = distribution.log_prob(action_sample).item()
        
        return action_index, log_prob
            
    def crop_input_and_cache(self, input_ids: torch.Tensor, past_key_values: DynamicCache, cache_position: int) -> tuple[torch.Tensor, DynamicCache, int]:
        if cache_position == 0:
            return input_ids, past_key_values, cache_position
        
        C = cache_position + 1
        past_key_values.crop(C)
        cropped_input_ids: torch.Tensor = input_ids.to(torch.int64)[:, C:]
        return cropped_input_ids, past_key_values, C
    
    def batched_inference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, input_lengths: list[int], past_key_values: torch.Tensor = None, training: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        with self.inference_lock:
            if training:
                self.model.train()
            else:
                self.model.eval()
                
            with torch.set_grad_enabled(training):
                output = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                    past_key_values=past_key_values,
                                    use_cache=not training,
                                    )
            last_prompt_token_indexes = np.array(input_lengths) - 1
            
            # get the values
            last_hidden_state = output['hidden_states'][-1]
            prompt_input_indexes = list(range(0, len(input_ids), self.process_batch_size))
            model_heads = last_hidden_state[prompt_input_indexes, last_prompt_token_indexes, :]
            model_heads = model_heads.to(torch.float32).to(self.device)
            
            with torch.set_grad_enabled(training):
                values = self.value_head_op.forward(model_heads)
            
            return torch.log_softmax(output.logits, dim=-1), values
       
