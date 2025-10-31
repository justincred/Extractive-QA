#!/bin/bash
#SBATCH --mem=24G
#SBATCH --gres=gpu:h200-141:1
#SBATCH -t 01:00:00
#SBATCH -J deberta_train
#SBATCH -o logs/out.out
nvidia-smi
python eval.py

