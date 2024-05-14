#!/bin/bash
source ~/.bashrc && conda activate inrdetect4

CUDA_VISIBLE_DEVICES=0 #,1,2,3
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=0, #1,2,3

MODEL=autoencoder
MODE=train
SCRIPT_NAME=${MODE}.py

CONFIG_DIR=configs
SPEC=16x16x16 # 8x8x64, 16x16x16, 32x32x4, 64x64x3
CONFIG_SPEC=${MODEL}_kl_${SPEC}

BASE_PATH=configs/autoencoder/may-14/three-stage.yaml
CHKPT_PATH=logs/2024-05-13T06-22-26_no_rec_0.7_dropout
echo "python $SCRIPT_NAME --base $BASE_PATH -r $CHKPT_PATH -t --gpus $GPUS"
python $SCRIPT_NAME -b $BASE_PATH -r $CHKPT_PATH -t --gpus $GPUS