#!/bin/bash
source ~/.bashrc && conda activate gen-detection

CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=0,1,2,3

MODE=train
MODEL=autoencoder
DATASET=shapenet
SCRIPT_NAME=${MODE}_${MODEL}.py

CONFIG_DIR=configs
SPEC=16x16x16 # 8x8x64, 16x16x16, 32x32x4, 64x64x3
CONFIG_SPEC=${MODEL}_kl_${SPEC}

echo "python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/$MODE_$DATASET/pose/$CONFIG_SPEC.yaml -t --gpus $GPUS"
python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/$MODE\_$DATASET/pose/$CONFIG_SPEC.yaml -t --gpus $GPUS