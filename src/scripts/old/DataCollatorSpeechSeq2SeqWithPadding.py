import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from abc import ABC, abstractmethod

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any = None
    text_name: str = "text"

    def __call__(self, batch: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        result = {}
        audio = [item["audio"]["array"] for item in batch]
        result["input_features"] = self.processor(
            audio, 
            return_tensors="pt", 
            sampling_rate=self.processor.feature_extractor.sampling_rate
            ).input_features
        
        labels = [item[self.text_name] for item in batch]
        labels_batch = self.processor.tokenizer(
            labels, 
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
            )
        result["labels"] = labels_batch.input_ids
        result["attention_mask"] = labels_batch.attention_mask

        return result
