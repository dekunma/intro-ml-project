#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=5:00:00                  
#SBATCH --mem=16GB
#SBATCH --job-name=pose_resnet152_2
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/pose_resnet152_2/pose_resnet152_2.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
# python train.py --model_name=pose_resnet152 --epoch=150 --mode=full
python train.py --config=./experiments/pose_resnet152/pose_resnet152_2.yaml --mode=full