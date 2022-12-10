#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=8:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=convnext1
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/convnext_1/convnext_1.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --config=./experiments/convnext/convnext_1.yaml --mode=full
# python train.py --model_name convnext --mode full --epoch 230