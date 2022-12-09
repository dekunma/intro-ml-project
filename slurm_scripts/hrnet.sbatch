#!/bin/bash
#SBATCH --cpus-per-task=2               
#SBATCH --time=6:00:00                  
#SBATCH --mem=16GB
#SBATCH --job-name=hrnet
#SBATCH --output=/scratch/dm4524/intro-ml-project/logs/hrnet_1/hrnet_1.out
#SBATCH --gres=gpu:rtx8000:1

cd /scratch/dm4524/intro-ml-project
conda activate cs473
# python train.py --model_name hrnet --mode full --epoch 150
python train.py --config=./experiments/hrnet/hrnet_1.yaml --mode=full