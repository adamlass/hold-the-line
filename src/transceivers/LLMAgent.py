from classes.Action import Action
from classes.ActionType import ActionType
from classes.Environment import Environment
from services.LLMService import LLMService
from transceivers.Transceiver import Transceiver
from colorama import Fore, Style
import torch
from transformers import DynamicCache

class LLMAgent(Transceiver):
    llm_service: LLMService
    
    # current model input
    generated_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache_position: int
    past_key_values: DynamicCache
    
    def __init__(self, environment: Environment):
        super().__init__("LLMAgent", environment, log_color=Fore.WHITE, wait_for_input=True, blocking=False, batch_processing=False, debug=False)
        self.llm_service = environment.llm_service
        
    def setup(self):
        if self.debug: self.log("Setting up system prompt")
        user_goal: str = self.environment.get_user_goal()
        system_prompt: str = self.llm_service.get_system_prompt(user_goal)
       
        self.generated_ids = torch.tensor([], device=self.llm_service.device, dtype=torch.int64)
        self.attention_mask = torch.tensor([], device=self.llm_service.device, dtype=torch.int64)
        self.cache_position = 0
        self.past_key_values = DynamicCache()
        
        chat = [
            {"role": "system", "content": system_prompt},
        ]
        inputs = self.llm_service.apply_chat_template(chat, add_generation_prompt=False)
        self.add_tokens(inputs.input_ids, inputs.attention_mask)
    
    def add_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, is_prompt: bool = True):
        self.generated_ids = torch.cat([self.generated_ids, input_ids], dim=-1)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=-1)
        
        if self.debug:
            decoded = self.llm_service.batch_decode(input_ids)[0]
            color = Fore.YELLOW if is_prompt else Fore.GREEN
            decoded_lines = decoded.split("\n")
            
            for decoded_line in decoded_lines:
                self.log(color + decoded_line + Style.RESET_ALL, flush=True, end="\n")
        
                
    def process(self, data):
        # self.log("Processing data", data)
        if data is None:
            return None
        
        action: Action = self.generate(data)
        
        return action
    
    def generate(self, chunk) -> Action:
        # adding the system prompt to the generated_ids
        chat = [{"role": "user", "content": chunk}]
        inputs = self.llm_service.apply_chat_template(chat, add_generation_prompt=True)
        # self.log("Generated input:", inputs.input_ids)
        self.add_tokens(inputs.input_ids, inputs.attention_mask, is_prompt=False)
        
        # generate the action and add it to the generated_ids
        action, new_past_key_values, new_cache_position = self.environment.prompt(self.generated_ids, self.attention_mask, self.past_key_values, self.cache_position)
        # self.log("Generated action:", action)
        self.past_key_values = new_past_key_values
        self.cache_position = new_cache_position
        # if self.llm_service.mode != "partial" or action.text != ActionType.wait:
        if True:
            action_token_ids = action.token_ids.unsqueeze(0).detach().to(torch.long).to(self.llm_service.device)
            #Append the end-of-input and a newline token to the action token IDs
            eoi_token_id = self.llm_service.tokenizer.eos_token_id
            end_tokens = torch.tensor([[eoi_token_id, self.llm_service.newline_token_id]], device=self.llm_service.device, dtype=torch.long)
            action_token_ids = torch.cat([action_token_ids, end_tokens], dim=-1)
            action_attention_mask = torch.ones_like(action_token_ids, device=self.llm_service.device, dtype=torch.float16)
            self.add_tokens(action_token_ids, action_attention_mask, is_prompt=False)
        
        return action
        
    def teardown(self):
        if self.debug: 
            self.log("Tearing down")
            generated_text = self.llm_service.batch_decode(self.generated_ids)
            self.log(generated_text)