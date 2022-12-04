#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=4:30:00                  
#SBATCH --mem=16GB
#SBATCH --job-name=resnet50
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/resnet50/resnet50.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --model_name resnet50 --mode full --epoch 150