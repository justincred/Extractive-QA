#!/bin/bash
#SBATCH --mem=24G
#SBATCH --gres=gpu:h200-141:1
#SBATCH -t 02:00:00
#SBATCH -J cs4248
#SBATCH -o logs/out_%j.out
nvidia-smi
# srun python preprocess.py --model_names microsoft/deberta-v3-base
srun python train_seed_ensemble.py --model_names microsoft/deberta-v3-base
srun python eval_seed_ensemble.py --model_names microsoft/deberta-v3-base
srun python evaluate-v2.0.py dev-v1.1.json predictions.json

