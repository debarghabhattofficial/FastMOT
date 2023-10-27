#!/bin/bash

# Use the following script to convert a ONNX model to the corresponding TensorRT engine.
# NOTE:
# --onnx: Path to the input ONNX model.
# --shapes=<input_name>:<input_shape>: Sets the input shape of the TensorRT engine output 'input_name' to 'input_shape'. IMPORTANT!
# --saveEngine: Path to the output TensorRT engine.
trtexec --onnx="../fastmot/models/osnet_x1_0_msdc.onnx" \
    --shapes=images:16x3x256x128 \
    --saveEngine="../fastmot/models/osnet_x1_0_msdc.trt" > "../fastmot/models/onnx2trt_conversion_logs/osnet_x1_0_msdc.txt"
