
from dataclasses import dataclass
from queue import Queue

@dataclass
class Call:
    id: str
    number: str
    audio_queue: Queue = None