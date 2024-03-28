#!/bin/bash
source ~/.bashrc && conda activate gen-detection

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=0,

MODEL=autoencoder
SCRIPT_NAME=${MODE}.py

CONFIG_DIR=configs
SPEC=16x16x16 # 8x8x64, 16x16x16, 32x32x4, 64x64x3
CONFIG_SPEC=${MODEL}_kl_${SPEC}

echo "python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/no-pose/$CONFIG_SPEC.yaml -t --gpus $GPUS"
python $SCRIPT_NAME --base $CONFIG_DIR/$MODEL/no-pose/$CONFIG_SPEC.yaml -t --gpus $GPUS