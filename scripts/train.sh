#!/bin/bash

# Change to ROOT directory.
cd ..

# Execute the python script.
python app.py \
    --input-uri "input/wobble.avi" \
    --config "cfg/mot-osnet_x0_25_msmt17.json" \
    --mot \
    --output-uri "results/code_test.mp4" \
    --txt "results/code_test.txt"





