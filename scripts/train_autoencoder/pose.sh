#!/bin/bash
source ~/.bashrc && conda activate gen-detection

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=0,

MODE=train
MODEL=autoencoder
DATASET=shapenet
SCRIPT_NAME=${MODE}_${MODEL}.py

CONFIG_DIR=configs
CONFIG_SPEC=${MODEL}_kl_8x8x64

echo "python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/$MODE_$DATASET/pose/$CONFIG_SPEC.yaml -t --gpus $GPUS"
python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/$MODE\_$DATASET/pose/$CONFIG_SPEC.yaml -t --gpus $GPUS