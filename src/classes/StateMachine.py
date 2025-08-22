from abc import ABC, abstractmethod

from classes.Episode import Episode


class StateMachine(ABC):
    episode: Episode
    done: bool
    frame: int
    
    def __init__(self, episode: Episode):
        self.episode = episode
        self.done = False
        self.frame = 0
        
    @abstractmethod
    def next_frame(self) -> str:
        """
        Returns the next frame of the episode.
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: str) -> tuple[float, bool]:
        """
        Applies the given action to the episode and returns the reward and whether the episode is done.
        """
        pass