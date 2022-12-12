#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=ray
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/ray_tune_resnest269/ray_tune_resnest269_2.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python ray_tune_resnest269.py