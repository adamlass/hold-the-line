

import random
from typing import List
from datasets import load_dataset, concatenate_datasets
import numpy as np
from torch.utils.data import DataLoader
import torch

from dataloading.SpeechDataset import SpeechDataset

RANDOM_SEED = 42

class SpeechDatasetLoader:
    def __init__(self, random_seed: int = RANDOM_SEED):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        self.g = torch.Generator()
        self.g.manual_seed(random_seed)

    def seed_worker(self, worker_id):
        worker_seed = RANDOM_SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def load_dataset(self, dataset_obj: SpeechDataset, streaming: bool = False, split=None, n: int = None):
        dataset = load_dataset(dataset_obj.path, dataset_obj.name, streaming=streaming, trust_remote_code=True, split=split)


        train_datasets = [dataset[split] for split in dataset_obj.train_splits]
        train_dataset = concatenate_datasets(train_datasets)
        
        eval_datasets = [dataset[split] for split in dataset_obj.eval_splits]
        eval_dataset = concatenate_datasets(eval_datasets)
        
        if n is not None:
            train_dataset = train_dataset.select(range(n))
            eval_dataset = eval_dataset.select(range(n))
        
        return train_dataset, eval_dataset

if __name__ == "__main__":
    from tqdm import tqdm

    tedlium = SpeechDataset("LIUM/tedlium", "release3", ignore_segment_tokens=["ignore_time_segment_in_scoring"])

    loader = SpeechDatasetLoader()
    train_loader, val_loader, test_loader = loader.load_dataset(tedlium)
    
    for batch in tqdm(train_loader):
        pass