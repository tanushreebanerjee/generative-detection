#!/bin/bash
source ~/.bashrc && conda activate inrdetect4

CUDA_VISIBLE_DEVICES=0 #,1,2,3
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
GPUS=0, #1,2,3

MODEL=autoencoder
MODE=train
SCRIPT_NAME=${MODE}.py

echo "python $SCRIPT_NAME -b configs/autoencoder/may17/aug_sc.yaml -t --gpus $GPUS"
srun python $SCRIPT_NAME -b configs/autoencoder/may17/aug_sc.yaml -t --gpus $GPUS