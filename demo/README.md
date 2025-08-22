# Episode Replay Demo Application

A web-based application for visualizing and replaying training episodes from the Hold The Line environment.

## Overview

This demo application provides an interactive interface to:
- Browse completed training runs
- View sample steps and their statistics
- Select and replay individual episodes
- Visualize observations, actions, and rewards in real-time

## Running the Demo

From the `demo` directory, run:

```bash
python app.py
```

Or with custom options:

```bash
python app.py --port 8000 --debug
```

### Command Line Options

- `--port`: Port to run the web server on (default: 5000)
- `--debug`: Run in debug mode for development

## Directory Structure

```
demo/
├── app.py              # Flask web server application
├── templates/
│   └── index.html      # Main HTML template
├── static/
│   ├── script.js       # Frontend JavaScript logic
│   └── style.css       # CSS styling
└── README.md           # This file
```

## Features

### Training Browser
- Lists all available training runs
- Shows metadata: episodes, sample steps, size, last modified
- Formatted display of training tags and timestamps

### Sample Step Selection
- View all sample steps for a training run
- See statistics per step:
  - Terminal successes
  - Total episodes
  - Success rate
  - Total samples collected

### Episode Selection
- Browse episodes for a specific sample step
- Filter by success/failure status
- View episode metadata:
  - Process and episode IDs
  - Sample count
  - Total reward
  - Success status

### Episode Playback
- Real-time replay of episode observations and actions
- Interactive progress bar with seeking
- Reward tracking and animations
- Play/pause and rewind controls
- Tooltips showing advantage, value, return, and reward values
- Visual distinction between action types and outcomes

## Data Requirements

The demo expects training data to be stored in `../data/trainings/` relative to the demo folder, with the following structure:

```
data/trainings/
└── training_tag_YYYYMMDD_HHMMSS/
    ├── metadata.json
    └── episodes/
        ├── process_0000_episode_0000/
        │   ├── metadata.json
        │   ├── sample_0000.json
        │   ├── sample_0001.json
        │   └── ...
        └── ...
```

This structure is automatically created when training with the `--save_samples` flag.

## Technical Details

- **Backend**: Flask web server with RESTful API endpoints
- **Frontend**: Vanilla JavaScript with modern CSS
- **Data Format**: JSON-based episode storage
- **Caching**: Episodes are cached in memory after first load for performance

## Browser Compatibility

Works best with modern browsers (Chrome, Firefox, Safari, Edge). Requires JavaScript enabled.
