from dataclasses import dataclass


@dataclass
class Goal:
    description: str
    correct_path_segment_ids: list[str]