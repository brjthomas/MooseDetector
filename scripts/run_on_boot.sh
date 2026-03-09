#!/bin/bash

PROJECT_DIR="/home/moose/projects/MooseDetector"
VENV_DIR="$PROJECT_DIR/venv"

#Activate virtual environment
source "$VENV_DIR/bin/activate"

#Navigate to project directory
cd "$PROJECT_DIR"

#Run the main.py application
python ./src/main.py
