#!/bin/bash
source ~/.bashrc && conda activate gen-detection

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=1

MODE=train
MODEL=autoencoder
DATASET=shapenet
SCRIPT_NAME=${MODE}_${MODEL}.py

CONFIG_DIR=configs
CONFIG_SPEC=${MODEL}_kl_8x8x64

echo "Running $SCRIPT_NAME with $CONFIG_DIR/$MODEL/$MODE_$DATASET/$CONFIG_SPEC.yaml"
python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/$MODE\_$DATASET/$CONFIG_SPEC.yaml -t --gpus $GPUS