
from dataclasses import dataclass
from typing import Any
from dataloading.DataCollatorSpeechSeq2SeqWithPadding import DataCollatorSpeechSeq2SeqWithPadding

class SpeechDataset:
    path: str
    name: str
    ignore_segment_tokens: list[str]
    text_name: str 
    train_splits: list[str]
    eval_splits: list[str]
    collator: DataCollatorSpeechSeq2SeqWithPadding
    
    def __init__(self, path: str, name: str, ignore_segment_tokens: list[str] = None, text_name: str = "text", train_splits: Any = ["train"], eval_splits: Any = ["test"]):
        self.path = path
        self.name = name
        self.ignore_segment_tokens = ignore_segment_tokens
        self.text_name = text_name
        self.train_splits = train_splits
        self.eval_splits = eval_splits
        
        self.collator = DataCollatorSpeechSeq2SeqWithPadding(
            text_name=text_name
        )