#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=5:00:00                  
#SBATCH --mem=16GB
#SBATCH --job-name=resnet152
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/resnet152/resnet152.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
python train.py --model_name resnet152 --mode=full --epoch=180
# python train.py --config=./experiments/resnet152/resnet152_1.yaml --mode=full