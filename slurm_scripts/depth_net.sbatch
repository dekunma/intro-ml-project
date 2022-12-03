#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=1:30:00             
#SBATCH --mem=16GB
#SBATCH --job-name=depth_net
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/depth_net/depth_net.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --model_name depth_net --mode full --epoch 100