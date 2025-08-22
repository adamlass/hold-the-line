from dataclasses import dataclass

from classes.Action import Action
import torch

@dataclass
class Sample:
    action: Action
    value: float
    reward: float
    prompt_ids: torch.Tensor
    next_done: bool
    log_prob: float
    advantage: float
    process_id: int
    returnn: float
    teacher_forced: bool
    observation_text: str = None
    user_goal: str = None
    call_tree_id: str = None
    
# pretty-print the dataclass
    def __repr__(self):
        return f"{self.action.text:<15} | p{str(self.process_id):<3} | {'terminal' if self.next_done else 'continued':<9} | {'teacher forced' if self.teacher_forced else 'organic':<14} | reward:{self.reward:7.4f} | value:{self.value:>16.16f} | adv: {self.advantage:>16.16f} | returnn: {self.returnn:>16.16f} | log_prob:{self.log_prob}"
    
    def to_dict(self):
        """Convert sample to dictionary for JSON serialization"""
        return {
            "action": self.action.text,
            "action_index": self.action.index,
            "value": float(self.value),
            "reward": float(self.reward),
            "next_done": self.next_done,
            "log_prob": float(self.log_prob),
            "advantage": float(self.advantage) if self.advantage is not None else None,
            "process_id": self.process_id,
            "returnn": float(self.returnn) if self.returnn is not None else None,
            "teacher_forced": self.teacher_forced,
            "observation_text": self.observation_text,
            "user_goal": self.user_goal,
            "call_tree_id": self.call_tree_id
        }
