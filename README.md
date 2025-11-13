# Instructions on how to run the code
## Running code for single models
1. Select a model from HuggingFace's list of NLP models e.g. "microsoft/deberta-v3-base".
2. In `preprocess.py`, set the `model_checkpoint` variable to the model from 1.
3. In `train.py` set the `model_name` argument of `train` to the model from 1.
4. In `eval.py`, set the `model_path` variable to the model from 1.
5. Run the script `run_train.sh` with the command `sbatch run_train.sh` in the SoC cluster.

## Running code for homogeneous ensemble models
1. Select a model from HuggingFace's list of NLP models e.g. "microsoft/deberta-v3-base".
2. In `run_train_seed_ensemble.sh`, replace all instances of `<model_name>` with the model from 1.
3. In `run_train_seed_ensmemble.sh`, replace all instances of `<seeds>` with comma-separated values of seeds e.g. "42,43,44".
4. Run the script `run_train_seed_ensemble.sh` with the command `sbatch run_train_seed_ensemble.sh` in the SoC cluster.

## Running code for heterogeneous ensemble models
1. Select several models from HuggingFace's list of NLP models e.g. "microsoft/deberta-v3-base,roberta-base".
2. In `run_train_ensemble.sh`, replace all instances of `<model_names>` with the comma-separated list of models from 1.
3. Run the script `run_train_ensemble.sh` with the command `sbatch run_train_ensemble.sh` in the SoC cluster.
