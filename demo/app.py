#!/usr/bin/env python3
"""
Web-based demo application for the new episode-based storage structure.
Handles episodes that span multiple sampling steps with proper temporal tracking.
"""

import json
import os
import time
import pickle
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
from typing import List, Dict, Any, Optional, Tuple
import argparse
import hashlib

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
# Global storage for loaded data
loaded_trainings = {}
MAX_CACHED_TRAININGS = 10  # Keep more trainings in memory since they're immutable


class CacheManager:
    """Manages caching of processed training data to disk."""
    
    @staticmethod
    def get_cache_path(training_dir: Path, cache_type: str) -> Path:
        """Get the path for a specific cache file."""
        cache_dir = training_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{cache_type}.pkl"
    
    @staticmethod
    def get_metadata_hash(metadata_file: Path) -> str:
        """Get hash of metadata file for cache invalidation."""
        if not metadata_file.exists():
            return ""
        
        # Use file modification time and size for quick hash
        stat = metadata_file.stat()
        hash_str = f"{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    @staticmethod
    def is_cache_valid(cache_path: Path, metadata_file: Path, max_age_seconds: Optional[int] = None) -> bool:
        """Check if cache is still valid.
        
        Args:
            cache_path: Path to the cache file
            metadata_file: Path to the metadata file to check against
            max_age_seconds: Maximum age in seconds (None = no expiration for immutable data)
        """
        if not cache_path.exists():
            return False
        
        # Only check cache age if max_age_seconds is specified
        # For immutable training data, we don't need time-based expiration
        if max_age_seconds is not None:
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > max_age_seconds:
                return False
        
        # Load cache and check metadata hash - this ensures cache is invalidated
        # only if the source data actually changes
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                cached_hash = cache_data.get('metadata_hash', '')
                current_hash = CacheManager.get_metadata_hash(metadata_file)
                return cached_hash == current_hash
        except:
            return False
    
    @staticmethod
    def load_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
        """Load cached data from disk."""
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    @staticmethod
    def save_cache(cache_path: Path, data: Dict[str, Any], metadata_file: Path):
        """Save data to cache with metadata hash."""
        try:
            cache_data = {
                'metadata_hash': CacheManager.get_metadata_hash(metadata_file),
                'timestamp': time.time(),
                'data': data
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")


class EpisodeReader:
    """Read and process episodes from the new storage structure with caching."""
    
    def __init__(self, training_dir: Path):
        """Initialize reader with a training directory."""
        self.training_dir = training_dir
        self.episodes_dir = training_dir / "episodes"
        self.metadata_file = training_dir / "metadata.json"
        self.cache_manager = CacheManager()
        
        # Load metadata with caching
        self.metadata = self._load_training_metadata()
        
        # Lazy loading - don't load episodes until needed
        self._episodes_cache = None
        self._step_episodes_cache = {}
        self._training_stats_cache = None
    
    def _load_training_metadata(self) -> Dict[str, Any]:
        """Load the training-level metadata with caching."""
        cache_path = self.cache_manager.get_cache_path(self.training_dir, "metadata")
        
        # Check if cache is valid
        if self.cache_manager.is_cache_valid(cache_path, self.metadata_file):
            cached = self.cache_manager.load_cache(cache_path)
            if cached:
                return cached['data']
        
        # Load fresh metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Save to cache
            self.cache_manager.save_cache(cache_path, metadata, self.metadata_file)
            return metadata
        return {}
    
    def _load_all_episodes(self) -> List[Dict[str, Any]]:
        """Load all episode metadata with disk caching."""
        if self._episodes_cache is None:
            cache_path = self.cache_manager.get_cache_path(self.training_dir, "episodes_list")
            
            # Try to load from disk cache first
            # No expiration for immutable training data
            if self.cache_manager.is_cache_valid(cache_path, self.metadata_file):
                cached = self.cache_manager.load_cache(cache_path)
                if cached:
                    self._episodes_cache = cached['data']
                    return self._episodes_cache
            
            # Load fresh data
            episodes = []
            if self.episodes_dir.exists():
                for episode_dir in sorted(self.episodes_dir.iterdir()):
                    if episode_dir.is_dir():
                        metadata_file = episode_dir / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                episode_data = json.load(f)
                                episode_data['episode_dir'] = str(episode_dir)
                                episodes.append(episode_data)
            
            # Save to disk cache
            self.cache_manager.save_cache(cache_path, episodes, self.metadata_file)
            self._episodes_cache = episodes
        
        return self._episodes_cache
    
    @property
    def episodes(self):
        """Lazy property for accessing episodes."""
        return self._load_all_episodes()
    
    def get_sample_steps(self) -> List[Dict[str, Any]]:
        """Get information about all sample steps from metadata only."""
        steps = []
        
        # Use only metadata - don't load episodes
        if 'sample_steps' in self.metadata:
            for step_info in self.metadata['sample_steps']:
                steps.append({
                    'step': step_info['step'],
                    'terminal_successes': step_info.get('terminal_successes', 0),
                    'total_episodes': step_info.get('episode_count', 0),
                    'total_samples': step_info.get('sample_count', 0)
                })
        
        return sorted(steps, key=lambda x: x['step'])
    
    def get_episodes_for_step(self, sample_step: int) -> List[Dict[str, Any]]:
        """Get all episodes active during a specific sample step with caching."""
        # Check memory cache first
        if sample_step in self._step_episodes_cache:
            return self._step_episodes_cache[sample_step]
        
        # Check disk cache
        cache_path = self.cache_manager.get_cache_path(self.training_dir, f"step_{sample_step}_episodes")
        # No expiration for immutable training data
        if self.cache_manager.is_cache_valid(cache_path, self.metadata_file):
            cached = self.cache_manager.load_cache(cache_path)
            if cached:
                active_episodes = cached['data']
                self._step_episodes_cache[sample_step] = active_episodes
                return active_episodes
        
        # Load fresh data
        active_episodes = []
        
        # Use the pre-loaded episodes list if available
        all_episodes = self._load_all_episodes()
        for episode_data in all_episodes:
            start_step = episode_data['start_sample_step']
            end_step = episode_data.get('end_sample_step', float('inf'))
            
            if start_step <= sample_step <= end_step:
                active_episodes.append(episode_data)
        
        # Save to disk cache
        self.cache_manager.save_cache(cache_path, active_episodes, self.metadata_file)
        
        # Cache in memory
        self._step_episodes_cache[sample_step] = active_episodes
        return active_episodes
    
    def load_episode_samples(self, process_id: int, episode_id: int) -> List[Dict[str, Any]]:
        """Load all samples for a specific episode."""
        episode_dir = self.episodes_dir / f"process_{process_id:04d}_episode_{episode_id:04d}"
        samples = []
        
        if episode_dir.exists():
            # Load all sample files in order
            sample_files = sorted(episode_dir.glob("sample_*.json"))
            for sample_file in sample_files:
                with open(sample_file, 'r') as f:
                    samples.append(json.load(f))
        
        return samples
    
    def get_episode_metadata(self, process_id: int, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific episode without loading all episodes."""
        episode_dir = self.episodes_dir / f"process_{process_id:04d}_episode_{episode_id:04d}"
        metadata_file = episode_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                episode_data = json.load(f)
                episode_data['episode_dir'] = str(episode_dir)
                return episode_data
        return None
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get cached training statistics."""
        if self._training_stats_cache is not None:
            return self._training_stats_cache
        
        # Check disk cache
        cache_path = self.cache_manager.get_cache_path(self.training_dir, "training_stats")
        # No expiration for immutable training data
        if self.cache_manager.is_cache_valid(cache_path, self.metadata_file):
            cached = self.cache_manager.load_cache(cache_path)
            if cached:
                self._training_stats_cache = cached['data']
                return self._training_stats_cache
        
        # Calculate fresh stats
        stats = {
            'episode_count': 0,
            'total_size': 0,
            'total_size_str': '0 KB'
        }
        
        if self.episodes_dir.exists():
            # Count episodes efficiently
            episode_dirs = [d for d in self.episodes_dir.iterdir() if d.is_dir()]
            stats['episode_count'] = len(episode_dirs)
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.training_dir.rglob('*') if f.is_file())
            stats['total_size'] = total_size
            stats['total_size_str'] = f"{total_size / (1024 * 1024):.1f} MB" if total_size > 1024 * 1024 else f"{total_size / 1024:.1f} KB"
        
        # Save to cache
        self.cache_manager.save_cache(cache_path, stats, self.metadata_file)
        self._training_stats_cache = stats
        return stats


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/trainings')
def get_trainings():
    """Get list of available training folders with caching."""
    training_list = []
    cache_manager = CacheManager()
    
    training_dir = Path('../data/trainings')
    if not training_dir.exists():
        return jsonify([])
    
    # Check for overall trainings list cache
    list_cache_path = training_dir / ".trainings_list_cache.pkl"
    
    # Check if we can use cached list
    # Use a longer cache time for the trainings list since new trainings are added infrequently
    cache_valid = False
    if list_cache_path.exists():
        cache_age = time.time() - list_cache_path.stat().st_mtime
        if cache_age < 3600:  # 1 hour cache for the list (can still manually clear if needed)
            try:
                with open(list_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    return jsonify(cached_data)
            except:
                pass
    
    for training_folder in training_dir.iterdir():
        if training_folder.is_dir() and training_folder.name.startswith('training_'):
            # Check if it has the new structure
            episodes_dir = training_folder / "episodes"
            metadata_file = training_folder / "metadata.json"
            
            if episodes_dir.exists() and metadata_file.exists():
                try:
                    # Use EpisodeReader with caching for stats
                    reader = EpisodeReader(training_folder)
                    stats = reader.get_training_stats()
                    
                    training_list.append({
                        'name': training_folder.name,
                        'folder': training_folder.name,
                        'size': stats['total_size_str'],
                        'modified': metadata_file.stat().st_mtime,
                        'episode_count': stats['episode_count'],
                        'sample_steps': len(reader.metadata.get('sample_steps', [])),
                        'tag': reader.metadata.get('training_tag', 'unnamed'),
                        'mode': reader.metadata.get('config', {}).get('mode', 'unknown')
                    })
                except Exception as e:
                    print(f"Error processing {training_folder.name}: {e}")
                    continue
    
    sorted_list = sorted(training_list, key=lambda x: x['modified'], reverse=True)
    
    # Save to cache
    try:
        with open(list_cache_path, 'wb') as f:
            pickle.dump(sorted_list, f)
    except:
        pass
    
    return jsonify(sorted_list)


@app.route('/api/sample_steps/<folder_name>')
def get_sample_steps(folder_name):
    """Get sample steps for a training folder."""
    global loaded_trainings
    
    training_dir = Path('../data/trainings') / folder_name
    if not training_dir.exists():
        return jsonify({'success': False, 'error': 'Training folder not found'})
    
    try:
        # Clear old cache if we have too many trainings loaded
        if len(loaded_trainings) >= MAX_CACHED_TRAININGS and folder_name not in loaded_trainings:
            # Remove the oldest accessed training (simple FIFO for now)
            oldest_key = next(iter(loaded_trainings))
            del loaded_trainings[oldest_key]
        
        # Load or get cached reader
        if folder_name not in loaded_trainings:
            loaded_trainings[folder_name] = EpisodeReader(training_dir)
        
        reader = loaded_trainings[folder_name]
        sample_steps = reader.get_sample_steps()
        
        return jsonify({
            'success': True,
            'sample_steps': sample_steps,
            'training_metadata': reader.metadata
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/episodes/<folder_name>/<int:sample_step>')
def get_episodes(folder_name, sample_step):
    """Get episodes for a specific sample step."""
    global loaded_trainings
    
    if folder_name not in loaded_trainings:
        training_dir = Path('../data/trainings') / folder_name
        if not training_dir.exists():
            return jsonify({'success': False, 'error': 'Training folder not found'})
        
        # Clear old cache if needed
        if len(loaded_trainings) >= MAX_CACHED_TRAININGS:
            oldest_key = next(iter(loaded_trainings))
            del loaded_trainings[oldest_key]
        
        loaded_trainings[folder_name] = EpisodeReader(training_dir)
    
    reader = loaded_trainings[folder_name]
    episodes = reader.get_episodes_for_step(sample_step)
    
    # Calculate statistics
    total_episodes = len(episodes)
    successful_episodes = sum(1 for e in episodes if e['success'])
    terminal_accuracy = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
    total_samples = sum(e['total_samples'] for e in episodes)
    
    return jsonify({
        'success': True,
        'episodes': episodes,
        'statistics': {
            'total_episodes': total_episodes,
            'successful_episodes': successful_episodes,
            'terminal_accuracy': f"{terminal_accuracy:.1f}%",
            'total_samples': total_samples
        }
    })


@app.route('/api/episode/<folder_name>/<int:process_id>/<int:episode_id>')
def get_episode_data(folder_name, process_id, episode_id):
    """Get the sample data for a specific episode."""
    global loaded_trainings
    
    if folder_name not in loaded_trainings:
        training_dir = Path('../data/trainings') / folder_name
        if not training_dir.exists():
            return jsonify({'success': False, 'error': 'Training folder not found'})
        
        # Clear old cache if needed
        if len(loaded_trainings) >= MAX_CACHED_TRAININGS:
            oldest_key = next(iter(loaded_trainings))
            del loaded_trainings[oldest_key]
        
        loaded_trainings[folder_name] = EpisodeReader(training_dir)
    
    reader = loaded_trainings[folder_name]
    samples = reader.load_episode_samples(process_id, episode_id)
    
    # Get episode metadata directly without loading all episodes
    episode_meta = reader.get_episode_metadata(process_id, episode_id)
    
    return jsonify({
        'success': True,
        'data': samples,
        'metadata': episode_meta
    })


@app.route('/api/clear_cache')
def clear_cache():
    """Clear all caches (memory and disk)."""
    global loaded_trainings
    loaded_trainings.clear()
    
    # Clear disk caches
    training_dir = Path('../data/trainings')
    if training_dir.exists():
        # Clear main list cache
        list_cache = training_dir / ".trainings_list_cache.pkl"
        if list_cache.exists():
            list_cache.unlink()
        
        # Clear individual training caches
        for training_folder in training_dir.iterdir():
            if training_folder.is_dir():
                cache_dir = training_folder / ".cache"
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.pkl"):
                        try:
                            cache_file.unlink()
                        except:
                            pass
    
    return jsonify({'success': True, 'message': 'All caches cleared (memory and disk)'})


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


def main():
    parser = argparse.ArgumentParser(description="Web-based episode replay demo for new storage structure")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting web server on http://localhost:{args.port}")
    print("Open this URL in your browser to view the demo")
    print("\nThis demo works with the new episode-based storage structure.")
    print("Episodes are stored in individual folders with sample files.")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
