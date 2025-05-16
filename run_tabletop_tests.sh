#!/bin/bash

# Script to run the tabletop environment test 10 times
# Created: $(date)

# Activate the sapien conda environment
# eval "$(conda shell.bash hook)"
# conda activate 

# Set counter
COUNT=1

# Run the command 10 times
while [ $COUNT -le 10 ]
do
    echo "Running test $COUNT of 10..."
    # xvfb-run -a python env_tests/connect_test_universal_tabletop.py --use-pi0 --save-images --env-id Tabletop-Close-Door-v1 --use-camera base_camera
    xvfb-run -a python env_tests/run_hierarchical_vla.py --use-cogact --save-images --env-id Tabletop-Close-Door-v1 --use-which-external-camera "base_camera"

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Test $COUNT completed successfully."
    else
        echo "Test $COUNT failed with exit code $?."
    fi
    
    # Increment counter
    COUNT=$((COUNT + 1))
    
    # Add a small delay between runs
    sleep 2
done

echo "All 10 tests completed."
