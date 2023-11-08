#!/bin/bash

# Change to ROOT directory.
cd ..

# Execute the python script.
# NOTE:
# Use --output-uri <output_path> to save output video to specific location.
# Use --show to visulaize the output during testing.
python app.py \
    --input-uri "input/wobble.avi" \
    --config "cfg/mot-mot-osnet_x0_25_msmt17.json" \
    --mot \
    --output-uri "results/code_exp19.mp4" \
    --show





