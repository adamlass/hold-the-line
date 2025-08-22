#!/usr/bin/env python3
"""
Enhanced episode-based storage system for training samples.
Handles episodes that span multiple sampling steps and maintains proper temporal relationships.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    process_id: int
    episode_id: int
    start_sample_step: int
    end_sample_step: Optional[int]
    total_samples: int
    success: bool
    total_reward: float
    user_goal: Optional[str]
    call_tree_id: Optional[str]
    start_timestamp: str
    end_timestamp: Optional[str]
    sample_step_transitions: List[Dict[str, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetadata:
    """Metadata for the entire training run."""
    training_tag: Optional[str]
    start_time: str
    config: Dict[str, Any]
    sample_steps: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodeStorage:
    """Manages episode-based storage for training samples."""
    
    def __init__(self, training_dir: Path, training_tag: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the episode storage system.
        
        Args:
            training_dir: Root directory for this training run
            training_tag: Optional tag for the training run
            config: Training configuration
        """
        self.training_dir = training_dir
        self.episodes_dir = training_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Track active episodes (episodes that haven't terminated yet)
        self.active_episodes: Dict[int, Dict[str, Any]] = {}  # process_id -> episode data
        
        # Track completed episodes
        self.completed_episodes: List[EpisodeMetadata] = []
        
        # Track sample step statistics
        self.sample_step_stats: Dict[int, Dict[str, Any]] = {}
        
        # Current sample step
        self.current_sample_step = 0
        
        # Episode counter per process
        self.episode_counters: Dict[int, int] = {}
        
        # Initialize training metadata
        self.training_metadata = TrainingMetadata(
            training_tag=training_tag,
            start_time=datetime.now().isoformat(),
            config=config or {},
            sample_steps=[]
        )
        
        # Save initial metadata
        self.save_training_metadata()
    
    def get_episode_dir(self, process_id: int, episode_id: int) -> Path:
        """Get the directory path for a specific episode."""
        episode_name = f"process_{process_id:04d}_episode_{episode_id:04d}"
        return self.episodes_dir / episode_name
    
    def start_episode(self, process_id: int, user_goal: Optional[str] = None, 
                     call_tree_id: Optional[str] = None):
        """
        Start tracking a new episode.
        
        Args:
            process_id: Process ID
            user_goal: User goal for this episode
            call_tree_id: Call tree ID for this episode
        """
        # Get episode ID for this process
        if process_id not in self.episode_counters:
            self.episode_counters[process_id] = 0
        episode_id = self.episode_counters[process_id]
        
        # Create episode directory
        episode_dir = self.get_episode_dir(process_id, episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize episode tracking
        self.active_episodes[process_id] = {
            'episode_id': episode_id,
            'episode_dir': episode_dir,
            'samples': [],
            'sample_count': 0,
            'total_reward': 0.0,
            'user_goal': user_goal,
            'call_tree_id': call_tree_id,
            'start_sample_step': self.current_sample_step,
            'start_timestamp': datetime.now().isoformat(),
            'sample_step_transitions': []
        }
        
        # Increment episode counter for next episode
        self.episode_counters[process_id] += 1
    
    def add_sample(self, process_id: int, sample_data: Dict[str, Any], is_extra_frame: bool = False):
        """
        Add a sample to an active episode.
        
        Args:
            process_id: Process ID
            sample_data: Sample data dictionary
            is_extra_frame: Whether this is an extra frame collected after episode end
        """
        if process_id not in self.active_episodes:
            # Start a new episode if not already active
            self.start_episode(
                process_id, 
                sample_data.get('user_goal'),
                sample_data.get('call_tree_id')
            )
        
        episode_data = self.active_episodes[process_id]
        sample_idx = episode_data['sample_count']
        
        # Check if we've transitioned to a new sample step
        if self.current_sample_step > episode_data.get('last_sample_step', episode_data['start_sample_step']):
            episode_data['sample_step_transitions'].append({
                'sample_idx': sample_idx,
                'from_step': episode_data.get('last_sample_step', episode_data['start_sample_step']),
                'to_step': self.current_sample_step
            })
        episode_data['last_sample_step'] = self.current_sample_step
        
        # Prepare sample for saving
        sample = {
            'sample_idx': sample_idx,
            'sample_step': self.current_sample_step,
            'timestamp': datetime.now().isoformat(),
            'is_extra_frame': is_extra_frame,
            **sample_data
        }
        
        # Save sample to file immediately
        sample_file = episode_data['episode_dir'] / f"sample_{sample_idx:04d}.json"
        with open(sample_file, 'w') as f:
            json.dump(sample, f, indent=2, default=str)
        
        # Update episode tracking
        episode_data['samples'].append(sample_file)
        episode_data['sample_count'] += 1
        episode_data['total_reward'] += sample_data.get('reward', 0)
        
        # Check if episode is done (and not an extra frame)
        if sample_data.get('next_done', False) and not is_extra_frame:
            self.end_episode(process_id, success=sample_data.get('reward', 0) == 1.0)
    
    def end_episode(self, process_id: int, success: bool = False):
        """
        End an active episode and save its metadata.
        
        Args:
            process_id: Process ID
            success: Whether the episode was successful
        """
        if process_id not in self.active_episodes:
            return
        
        episode_data = self.active_episodes[process_id]
        
        # Create episode metadata
        metadata = EpisodeMetadata(
            process_id=process_id,
            episode_id=episode_data['episode_id'],
            start_sample_step=episode_data['start_sample_step'],
            end_sample_step=self.current_sample_step,
            total_samples=episode_data['sample_count'],
            success=success,
            total_reward=episode_data['total_reward'],
            user_goal=episode_data['user_goal'],
            call_tree_id=episode_data['call_tree_id'],
            start_timestamp=episode_data['start_timestamp'],
            end_timestamp=datetime.now().isoformat(),
            sample_step_transitions=episode_data['sample_step_transitions']
        )
        
        # Save episode metadata
        metadata_file = episode_data['episode_dir'] / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Track completed episode
        self.completed_episodes.append(metadata)
        
        # Update sample step statistics
        if self.current_sample_step not in self.sample_step_stats:
            self.sample_step_stats[self.current_sample_step] = {
                'episode_count': 0,
                'sample_count': 0,
                'terminal_successes': 0
            }
        
        stats = self.sample_step_stats[self.current_sample_step]
        stats['episode_count'] += 1
        stats['sample_count'] += episode_data['sample_count']
        if success:
            stats['terminal_successes'] += 1
        
        # Remove from active episodes
        del self.active_episodes[process_id]
    
    def increment_sample_step(self):
        """Move to the next sample step."""
        self.current_sample_step += 1
        
        # Update training metadata with sample step statistics
        if self.current_sample_step - 1 in self.sample_step_stats:
            step_stats = self.sample_step_stats[self.current_sample_step - 1]
            self.training_metadata.sample_steps.append({
                'step': self.current_sample_step - 1,
                **step_stats
            })
            self.save_training_metadata()
    
    def save_training_metadata(self):
        """Save the training metadata to file."""
        metadata_file = self.training_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.training_metadata.to_dict(), f, indent=2)
    
    def finalize(self):
        """
        Finalize storage, ensuring all episodes are properly closed.
        """
        # End any remaining active episodes
        for process_id in list(self.active_episodes.keys()):
            self.end_episode(process_id, success=False)
        
        # Final metadata save
        self.save_training_metadata()
        
        print(f"Storage finalized: {len(self.completed_episodes)} episodes saved")
    
    def get_episode_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all episodes with their metadata.
        
        Returns:
            List of episode metadata dictionaries
        """
        episodes = []
        
        # Read from episode directories
        for episode_dir in sorted(self.episodes_dir.iterdir()):
            if episode_dir.is_dir():
                metadata_file = episode_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        episodes.append(json.load(f))
        
        return episodes
    
    def get_episodes_by_sample_step(self, sample_step: int) -> List[Dict[str, Any]]:
        """
        Get all episodes that were active during a specific sample step.
        
        Args:
            sample_step: The sample step to filter by
            
        Returns:
            List of episode metadata for episodes active during the sample step
        """
        episodes = []
        
        for episode in self.get_episode_list():
            start_step = episode['start_sample_step']
            end_step = episode.get('end_sample_step', float('inf'))
            
            # Check if episode was active during this sample step
            if start_step <= sample_step <= end_step:
                episodes.append(episode)
        
        return episodes
