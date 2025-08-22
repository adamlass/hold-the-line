
import copy
import json
import os
from queue import Empty, Queue
from random import shuffle
import random
from threading import Lock
from typing import Iterator
from classes import Segment, SegmentType
from classes.Episode import Episode
from classes.Goal import Goal


PROCESSED_DATA_FOLDER = "data_processed"
TREES_PATH = f"{PROCESSED_DATA_FOLDER}/trees"

class EpisodeLoader:
    __all_episodes: list[Episode] = []
    episode_queue: Queue[Episode] = Queue()
    lock: Lock = Lock()
    
    def __init__(self):
        random.seed(42)  # For reproducibility
        # get list of directories in TREES_PATH
        tree_names = self.get_directories(TREES_PATH)
        print(f"Found {len(tree_names)} trees")
        tree_names.sort(key=lambda x: int(x.split("_")[1]))
        
        for tree_name in tree_names:
            try:
                tree_file_path = f"{TREES_PATH}/{tree_name}/callTreeReplacedWithGoals.json"
                tree: Segment = json.load(open(tree_file_path))
                self.set_episodes(tree, tree_name)
            except FileNotFoundError:
                print(f"File not found: {tree_file_path}")
                continue
            
        print(f"Loaded {len(self.__all_episodes)} episodes from {len(tree_names)} trees.")  
        
        # Shuffle the episodes
        shuffle(self.__all_episodes)
        
        # Add episodes to the queue
        for episode in self.__all_episodes:
            self.episode_queue.put(episode)
    
    def next_episode(self) -> Episode:
        """
        Generates a new unused episode.
        """
        try:
            with self.lock:
                episode = self.episode_queue.get()
                self.episode_queue.put(episode)
                return episode
        except Empty:
            print("Episode queue is empty!")
            return None
        
    def get_directories(self, path):
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    def set_episodes(self, segment: Segment, tree_name: str, path_segment_ids: list[str] = []):
        current_segment_ids = path_segment_ids + [segment["id"]]
        
        if segment["navigationGoals"] is not None:
            for goal_description in segment["navigationGoals"]:
                goal = Goal(goal_description, current_segment_ids)
                episode = Episode(tree_name, goal)
                with self.lock:
                    self.__all_episodes.append(episode)
        
        if segment["type"] == SegmentType.dialOptions:
            for i, content in enumerate(segment["contents"]):
                # if i == 0:
                #     # Skip the first content
                #     continue
                self.set_episodes(content["nextSegment"], tree_name, current_segment_ids)
                    
        if segment["nextSegment"] is not None:
            self.set_episodes(segment["nextSegment"], tree_name, current_segment_ids)
    
if __name__ == "__main__":
    loader = EpisodeLoader()
    episode = loader.next_episode()
    print(episode)