
from models.BaseModel import BaseModel
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_constant_schedule_with_warmup
import torch
from torch.utils.data import DataLoader

class WhisperModel(BaseModel):
    def evaluate(self, dataloader):
        raise NotImplementedError

    def predict(self, dataloader):
        raise NotImplementedError

    def augment(self, **kwargs):
        raise NotImplementedError

    def load_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        self.processor = WhisperProcessor.from_pretrained(self.pretrained_model_name)


    
