#!/bin/bash

# Use the following script to convert a ONNX model to the corresponding TensorRT engine.
# NOTE:
# --onnx: Path to the input ONNX model.
# --shapes=<input_name>:<input_shape>: Sets the input shape of the TensorRT engine output 'input_name' to 'input_shape'. IMPORTANT!
# --saveEngine: Path to the output TensorRT engine.
#  trtexec --onnx="../fastmot/models/yolo_v7_3_1.onnx" \
#      --saveEngine="../fastmot/models/yolo_v7_3_1.trt" > "../fastmot/models/onnx2trt_conversion_logs/yolo_v7_3_1.txt"

/usr/local/TensorRT-8.5.1.7/bin/trtexec --onnx="../fastmot/models/resnet50_fc512_msmt17.onnx" \
    --shapes=images:16x3x256x128 \
    --saveEngine="../fastmot/models/resnet50_fc512_msmt17.trt"  # "../fastmot/models/onnx2trt_conversion_logs/osnet_ain_x1_0_msdc.txt"
