#!/bin/bash

# Change to ROOT directory.
cd ..

# Execute the python script.
python app.py \
    --input-uri "input/wobble.avi" \
    --config "cfg/mot-monarch_default.json" \
    --mot \
    --output-uri "results/input_wobble-config_mot_monarch_default-fr_100ms.mp4" \
    --txt "results/input_wobble-config_mot_monarch_default-fr_100ms.txt" \
    --show


