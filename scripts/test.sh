#!/bin/bash

# Change to ROOT directory.
cd ..

# Execute the python script.
# NOTE:
# Use --output-uri <output_path> to save output video to specific location.
# Use --show to visulaize the output during testing.
python app2.py \
    --input-uri "input/shadow_debugging.mp4" \
    --config "cfg/mot-osnet_x0_25_msmt17.json" \
    --output-uri "results/test-sd1-code_creation.mp4" \
    --mot


# python app2.py \
#     --input-uri "input/shadow_debugging.mp4" \
#     --config "cfg/mot-osnet_x0_25_msmt17.json" \
#     --mot \
#     --output-uri "results/test-sd1-new2.mp4" \
#     --show




