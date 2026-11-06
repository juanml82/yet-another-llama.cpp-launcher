#!/bin/bash

# 1. Navigate to the directory
echo "Changing directory to /mnt/shared/ai_apps/llama-launcher-mio..."
cd /mnt/shared/ai_apps/llama-launcher-mio || { echo "Directory not found"; exit 1; }

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source ./venv/bin/activate

# 3. Run the Python script
echo "Starting app.py..."
python app.py
