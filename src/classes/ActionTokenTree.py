
from dataclasses import dataclass

@dataclass
class ActionTokenTree:
    token_id: int
    children: list["ActionTokenTree"]