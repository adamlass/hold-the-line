
from abc import ABC, abstractmethod
from dataloading import DataCollatorSpeechSeq2SeqWithPadding
from dataloading.SpeechDataset import SpeechDataset
from dataloading.SpeechDatasetLoader import SpeechDatasetLoader
import evaluate
from models.ModelIdentifier import ModelIdentifier
import torch
from torch.utils.data import DataLoader
from utils import get_best_device
import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import os
import wandb.util
from dotenv import load_dotenv

load_dotenv()

WANDB_PROJECT_NAME = os.getenv("WANDB_PROJECT_NAME")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
MODEL_DIR = os.getenv("MODEL_DIR") or "./data/models"

class BaseModel(ABC):
    device = None
    model = None

    def __init__(self, pretrained_model_name:str, device: torch.device = get_best_device()):
        self.pretrained_model_name = pretrained_model_name
        self.device = device

    def new_training(self, config: dict = None):
        wandb.init(project=WANDB_PROJECT_NAME,
                    entity=WANDB_ENTITY,
                    resume="allow",
                    config=config)
        return wandb.run.name

    def resume_training(self, model_identifier: ModelIdentifier):
        wandb.init(project=WANDB_PROJECT_NAME,
                    entity=WANDB_ENTITY,
                    resume="must",
                    id=model_identifier.run_id)
        
        return wandb.run.name

    def send_to_device(self):
        if self.model is not None:
            print(f"Sending model to device: ", self.device)
            self.model.to(self.device)
            print("Model sent to device")
        else:
            raise Exception("Model is not initialized")
    
    def log_metrics(self, metrics: dict):
        wandb.log(metrics)
        
    def add_tag(self, tag: str):
        wandb.run.tags =wandb.run.tags + (tag,)

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False, normalize=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True, normalize=True)
        
        # skip empty labels
        skip_ids = [i for i in range(len(label_str)) if label_str[i] == ""]
        pred_str = [pred_str[i] for i in range(len(pred_str)) if i not in skip_ids]
        label_str = [label_str[i] for i in range(len(label_str)) if i not in skip_ids]

        wer = 100 * self.wer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def train(self, 
              dataset:SpeechDataset, 
              epochs=1, 
              resume_model: ModelIdentifier = None, 
              n: int=None, 
              batch_size: int=16,
              learning_rate: float=1e-5,
              warmup_steps: int=500,
              save_steps: int=1000,
              eval_steps: int=1000,
              logging_steps: int=25,
              save_total_limit: int=2,
              eval_on_start: bool=True):
        
        self.wer = evaluate.load("wer")
        
        loader = SpeechDatasetLoader()
        train_dataset, eval_dataset = loader.load_dataset(dataset, n=n)
        
        dataset.collator.processor = self.processor
        
        if resume_model is not None:
            run_id = self.resume_training(resume_model)
            print(f"Resuming training from run {run_id}")
        else:
            run_id = self.new_training()
            print(f"Starting new training run {run_id}")
            
            self.add_tag(f"{dataset.path}:{dataset.name}")
            
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"{MODEL_DIR}/{run_id}",  # change to a repo name of your choice
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            gradient_checkpointing=True,
            fp16=self.device=="cuda", # set to True if your GPU supports it
            evaluation_strategy="steps",
            per_device_eval_batch_size=8, # TODO check up on best value
            predict_with_generate=True,
            generation_max_length=225,
            save_total_limit=save_total_limit,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_on_start=eval_on_start,
            logging_steps=logging_steps,
            report_to=["wandb"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False
        )
        
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens': dataset.ignore_segment_tokens}, replace_additional_special_tokens=False)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        
        self.send_to_device()
    
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=dataset.collator,
            compute_metrics=self.compute_metrics,
        )
        
        if resume_model is None:
            trainer.train()
        else:
            model_path=f"{MODEL_DIR}/{run_id}/checkpoint-{resume_model.step}"
            trainer.train(resume_from_checkpoint=model_path)
        
        wandb.finish()
    
    @abstractmethod
    def load_model(self):
        raise NotImplementedError("Load model method not implemented")

    @abstractmethod
    def evaluate(self, dataloader: DataLoader):
        raise NotImplementedError("Evaluate method not implemented")

    @abstractmethod
    def predict(self, dataloader: DataLoader):
        raise NotImplementedError("Predict method not implemented")

    @abstractmethod
    def augment(self, **kwargs):
        raise NotImplementedError("Augment method not implemented")
    
