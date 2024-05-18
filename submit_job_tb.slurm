#!/bin/bash
#SBATCH --job-name=clip_nan_trace_od_vae_full_4gpu
#SBATCH --output=slurm_output/clip_nan_trace_od_vae_full_4gpu_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -A pci
#SBATCH -w node300
#SBATCH --time=2:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tb21@princeton.edu

source ~/.bashrc
conda activate inrdetect4

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python train.py -b configs/autoencoder/pose/jo_autoencoder_kl_16x16x16.yaml -t --name clip_nan_trace_od_vae_full_4gpu --devices 4