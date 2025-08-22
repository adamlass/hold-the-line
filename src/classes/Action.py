
from dataclasses import dataclass

import torch

@dataclass
class Action:
    index: int
    text: str
    token_ids: torch.Tensor