#!/bin/bash

iteration=0

while true
do
    # Increment the iteration counter
    clear
    iteration=$((iteration+1))

    # Print the iteration number and time
    echo "Iteration $iteration: $(date)"

    # Kill the previous instance of the script
    pkill -f "snake.py"

    # Run the Python script in the background
    python snake.py &

    # Sleep for 5 minutes
    sleep 180
done