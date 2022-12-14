#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=resnest269_9
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/resnest269_9/resnest269_9.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --config=./experiments/resnest269/resnest269_9.yaml --mode=full