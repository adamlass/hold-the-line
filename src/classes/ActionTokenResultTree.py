
from dataclasses import dataclass
from classes.ActionTokenTree import ActionTokenTree

@dataclass
class ActionTokenResultTree(ActionTokenTree):
    probability: float = None