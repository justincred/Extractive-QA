#!/bin/bash
#SBATCH --mem=24G
#SBATCH --gres=gpu:h200-141:1
#SBATCH -t 01:00:00
#SBATCH -J deberta_train
#SBATCH -o logs/out.out
nvidia-smi
# srun python preprocess.py
srun python train.py
srun python eval.py
srun python evaluate-v2.0.py dev-v1.1.json predictions.json

