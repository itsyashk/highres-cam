#!/bin/bash
# Forward SIGINT/SIGTERM to stop uvicorn gracefully
trap 'echo "Stopping server..."; pkill -f "uvicorn.*web_backend"; exit' SIGINT SIGTERM

# Kill any existing uvicorn processes
echo "Stopping any existing servers..."
pkill -f "uvicorn.*web_backend" 2>/dev/null
pkill -f "python.*vimba" 2>/dev/null

# Wait a moment for processes to clean up
sleep 2

# Check if any processes are still running
if pgrep -f "uvicorn.*web_backend" > /dev/null; then
    echo "Warning: Some processes may still be running"
    ps aux | grep -E "(uvicorn|vimba)" | grep -v grep
fi

# Activate virtual environment and start server
echo "Starting Vimba Centroid Lab web server..."
source venv/bin/activate
python -m uvicorn vimba_centroid_lab.web_backend:app --host 0.0.0.0 --port 8000
