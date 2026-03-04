#!/bin/bash

# Define the script name
SCRIPT_NAME="agent-mcp.py"

# Find the PID(s) of the running script
# Using pgrep -f to match the full command line
PIDS=$(pgrep -f "$SCRIPT_NAME")

if [ -z "$PIDS" ]; then
    echo "No running instance of $SCRIPT_NAME found."
else
    echo "Found $SCRIPT_NAME running with PID(s): $PIDS. Killing now..."
    # Kill the processes
    kill $PIDS
    # Give the OS a moment to release the ports/resources
    sleep 1
fi

# Restart the process
echo "Starting $SCRIPT_NAME..."
nohup python "$SCRIPT_NAME" >> agent-mcp.log 2>&1 &

echo "Process started. PID: $!"
