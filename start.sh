#!/bin/bash

# Navigate to your project directory
cd /home/ec2-user/flight_price_predition || exit 1

# Activate virtual environment
source venv/bin/activate

# Install required Python packages (optional, useful if dependencies might change)
pip install -r requirements.txt

# Kill any running Gunicorn processes (ignore errors if none running)
pkill gunicorn || true

# Start the Gunicorn server in the background
gunicorn --bind 0.0.0.0:5000 app:app &

