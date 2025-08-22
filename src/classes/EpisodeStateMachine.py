
import json
import random

from classes import Content, Segment, SegmentType
from classes.Action import Action
from classes.ActionType import ActionType
from classes.StateMachine import StateMachine
from random import shuffle

PROCESSED_DATA_FOLDER = "data_processed"
TREES_PATH = f"{PROCESSED_DATA_FOLDER}/trees"
CHUNK_JOIN_MIN = 3
CHUNK_JOIN_MAX = 4

class EpisodeStateMachine(StateMachine):
    mode: str
    current_segment: Segment
    silence: bool
    content_index: int
    chunk_index: int
    chunk_join_range: range
    current_content: Content
    current_chunk: str
    
    def __init__(self, episode, mode: str = "full"):
        """
        Initializes the EpisodeStateMachine with the given episode.
        """
        super().__init__(episode)
        self.mode = mode
        self.file_path = f"{TREES_PATH}/{episode.call_tree_id}/callTreeReplacedWithGoals.json"
        self.call_tree: Segment = json.load(open(self.file_path))
        self.change_call_tree_symbols(self.call_tree)
        if self.mode == "partial":
            self.chunk_join_range = range(CHUNK_JOIN_MIN, CHUNK_JOIN_MAX + 1)
            self.create_partial_observations(self.call_tree)
            # self.current_segment = self.call_tree["nextSegment"]
        #     current_contents = self.current_segment["contents"]
        #     for content in current_contents:
        #         next_segment_id = content["nextSegment"]["id"]
        #         if next_segment_id in self.episode.goal.correct_path_segment_ids:
        #             self.current_segment = content["nextSegment"]
        #             break
        # else:
        #     self.current_segment = self.call_tree
        self.current_segment = self.call_tree
        self.content_index = 0
        self.chunk_index = 0
        self.silence = False
        self.reward_scaling_factor = 0.1 if self.mode == "partial" else 1.0
        
    def change_call_tree_symbols(self, current_segment: Segment) -> Segment:
        if current_segment["type"] == SegmentType.welcome:
            self.change_call_tree_symbols(current_segment["nextSegment"])
            
        if current_segment["type"] == SegmentType.dialOptions:
            new_indexes = list(range(1, 10))
            shuffle(new_indexes)
            for i, content in enumerate(current_segment["contents"]):
                index = content["index"]
                new_index = new_indexes.pop(0)
                content["index"] = new_index
                content["text"] = content["text"].replace(f"{index}", f"{new_index}")
                
                self.change_call_tree_symbols(content["nextSegment"])
    
    def create_partial_observations(self, current_segment: Segment) -> Segment:
        for content in current_segment["contents"]:
            all_chunks = content["text"].split(" ")
            new_chunks = []
            
            i = 0
            while i < len(all_chunks):
                num_chunks_random = random.choice(self.chunk_join_range)
                new_chunks.append(" ".join(all_chunks[i:i + num_chunks_random]))
                i += num_chunks_random
            
            content["text_chunks"] = new_chunks
            if current_segment["type"] == SegmentType.dialOptions and content["nextSegment"]:
                self.create_partial_observations(content["nextSegment"])
                
        if current_segment["nextSegment"]:
            self.create_partial_observations(current_segment["nextSegment"])
    
    def get_reward_scale(self) -> float:
        """
        Scales the reward based on the mode.
        """
        if self.mode == "partial":
            current_content_words = self.current_content["text"].split(" ")
            current_chunk_words = self.current_chunk.split(" ")
            resolution = len(current_chunk_words) / len(current_content_words)
            return resolution
        return 1.0
    
    def next_frame(self) -> str:
        if self.mode == "full":
            return self.next_frame_full_observations()
        elif self.mode == "partial":
            return self.next_frame_partial_observations()
        elif self.mode == "complete":
            return self.next_frame_complete()
    
    def next_frame_full_observations(self) -> str:
        """
        Returns the next frame of the episode with full observations.
        """
        if self.content_index >= len(self.current_segment["contents"]):
            if self.current_segment["nextSegment"]:
                self.current_segment = self.current_segment["nextSegment"]
                self.content_index = 0
                self.chunk_index = 0
                self.silence = False
            else:
                self.silence = True
                return "[SILENCE]"
        
        current_content: Content = self.current_segment["contents"][self.content_index]
        self.content_index += 1
        
        return current_content["text"]
    
    def next_frame_partial_observations(self) -> str:
        """
        Returns the next frame of the episode with partial observations.
        """
        if self.content_index >= len(self.current_segment["contents"]):
            if self.current_segment["nextSegment"]:
                self.current_segment = self.current_segment["nextSegment"]
                self.content_index = 0
                self.chunk_index = 0
                self.silence = False
            else:
                self.silence = True
                return "[SILENCE]"
        
        self.current_content: Content = self.current_segment["contents"][self.content_index]
        self.current_chunk = self.current_content["text_chunks"][self.chunk_index]
        self.chunk_index += 1
        
        if self.chunk_index >= len(self.current_content["text_chunks"]):
            self.chunk_index = 0
            self.content_index += 1
            
        return self.current_chunk
        
    
    def next_frame_complete(self) -> str:
        """
        Seeing all options
        """
        results = []
        if self.current_segment["nextSegment"]:
            results.append(self.current_segment["contents"][self.content_index]["text"])
            self.current_segment = self.current_segment["nextSegment"]
        
        if self.current_segment["type"] == SegmentType.dialOptions:
            for content in self.current_segment["contents"]:
                results.append(content["text"])
            results.append("[SILENCE]")
            self.silence = True
                
        return " ".join(results)
    
    def apply_action(self, action: Action) -> tuple[float, bool]:
        """
        Applies the given action to the episode and returns the reward and whether the episode is done.
        """
        reward = 0.0
        
        if self.current_segment["type"] == SegmentType.welcome:
            if action.text == ActionType.wait:
                reward = 0.5 * self.get_reward_scale()
            else:
                reward = -1.0 * self.get_reward_scale()
                # self.done = True
        elif self.current_segment["type"] == SegmentType.dialOptions:
            if "press" in action.text:
                press_index = int(action.text.split(" ")[-1])
                available_indices = [int(content["index"]) for content in self.current_segment["contents"]]
                if press_index in available_indices:
                    self.current_segment = self._get_press_segment(press_index)
                    self.content_index = 0
                    self.chunk_index = 0
                    segment_id = self.current_segment["id"]
                    if segment_id in self.episode.goal.correct_path_segment_ids:
                        if segment_id == self.episode.goal.correct_path_segment_ids[-1]:
                            reward = 1.0
                        else:
                            reward = 1.0
                    else:
                        self.done = True
                        reward = -1.0
                else:
                    self.done = True
                    reward = -1.0
            elif action.text == ActionType.wait:
                if self.silence:
                    reward = -0.5
                    self.done = True
                else:
                    reward = -0.1 * self.get_reward_scale()
                    
        if (self.current_segment["type"] == SegmentType.holdTheLine):
            self.done = True
        
        return reward, self.done
        
    def get_best_action_index(self) -> int:
        """
        Returns the best action index based on the current segment.
        """
        if self.current_segment["type"] == SegmentType.welcome:
            return 0
        
        if self.current_segment["type"] == SegmentType.dialOptions:
            for content in self.current_segment["contents"]:
                if content["nextSegment"]["id"] in self.episode.goal.correct_path_segment_ids:
                    return content["index"]
        
    def _get_press_segment(self, press_index: int) -> Segment:
        """
        Returns the segment corresponding to the given press index.
        """
        for content in self.current_segment["contents"]:
            if int(content["index"]) == press_index:
                return content["nextSegment"]
        
        return None
    
    
    
    # def next_frame(self) -> str:
    #     """
    #     Swap option order
    #     """
    #     if self.content_index >= len(self.current_segment["contents"]):
    #         if self.current_segment["nextSegment"]:
    #             self.current_segment = self.current_segment["nextSegment"]
    #             self.content_index = 0
    #             self.silence = False
    #         else:
    #             self.silence = True
    #             return "[SILENCE]"
        
    #     current_content: Content = self.current_segment["contents"][- (1 + self.content_index)]
    #     self.content_index += 1
        
    #     return current_content["text"]