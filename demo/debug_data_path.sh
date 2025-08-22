#!/bin/bash

# Debug script to check data path issues in Docker container

echo "==================================="
echo "Data Path Debugging for Whisper Demo"
echo "==================================="
echo

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
else
    echo "✗ Not running in Docker. Run this inside the container:"
    echo "  docker-compose exec whisper-demo bash debug_data_path.sh"
    exit 1
fi

echo
echo "Current working directory:"
pwd

echo
echo "Checking expected data path (../data/trainings):"
if [ -d "../data/trainings" ]; then
    echo "✓ Directory exists at ../data/trainings"
    echo "  Contents:"
    ls -la ../data/trainings/ 2>/dev/null | head -10
    echo "  Number of training folders: $(find ../data/trainings -maxdepth 1 -type d -name 'training_*' | wc -l)"
else
    echo "✗ Directory NOT found at ../data/trainings"
fi

echo
echo "Checking /data/trainings path:"
if [ -d "/data/trainings" ]; then
    echo "✓ Directory exists at /data/trainings"
    echo "  Contents:"
    ls -la /data/trainings/ 2>/dev/null | head -10
    echo "  Number of training folders: $(find /data/trainings -maxdepth 1 -type d -name 'training_*' | wc -l)"
else
    echo "✗ Directory NOT found at /data/trainings"
fi

echo
echo "Checking /app/data/trainings path:"
if [ -d "/app/data/trainings" ]; then
    echo "✓ Directory exists at /app/data/trainings"
    echo "  Contents:"
    ls -la /app/data/trainings/ 2>/dev/null | head -10
    echo "  Number of training folders: $(find /app/data/trainings -maxdepth 1 -type d -name 'training_*' | wc -l)"
else
    echo "✗ Directory NOT found at /app/data/trainings"
fi

echo
echo "All mounted volumes:"
df -h | grep -E "^/dev|Filesystem"

echo
echo "Environment check:"
echo "  FLASK_ENV=$FLASK_ENV"
echo "  PATH=$PATH"
echo "  PWD=$PWD"

echo
echo "Python path resolution test:"
python3 -c "
import os
from pathlib import Path
print('Python working directory:', os.getcwd())
training_dir = Path('../data/trainings')
print(f'Path ../data/trainings resolves to: {training_dir.resolve()}')
print(f'Path exists: {training_dir.exists()}')
if training_dir.exists():
    trainings = list(training_dir.glob('training_*'))
    print(f'Found {len(trainings)} training folders')
    for t in trainings[:3]:
        print(f'  - {t.name}')
"

echo
echo "==================================="
echo "Debugging complete!"
echo "===================================
"
