#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=resnest269_1
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/resnest269_1/resnest269_1.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
# python train.py --model_name resnest269 --mode full --epoch 230
python train.py --config=./experiments/resnest269/resnest269_1.yaml --mode=full