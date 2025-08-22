

from dataclasses import dataclass
from classes.Goal import Goal

@dataclass
class Episode:
    call_tree_id: str
    goal: Goal