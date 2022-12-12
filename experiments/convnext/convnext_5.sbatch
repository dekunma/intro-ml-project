#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=convnext5
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/convnext_5/convnext_5.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --config=./experiments/convnext/convnext_5.yaml --mode=head