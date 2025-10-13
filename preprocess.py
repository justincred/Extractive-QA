from datasets import load_dataset
from transformers import AutoTokenizer
dataset = load_dataset(
    "json",
    data_files={
        "train": "train-v1.1.json",
        "validation": "dev-v1.1.json"
    },
    field="data"  # top-level field to parse
)
print(dataset["train"])
print(dataset["train"][0])  # first example
print(dataset["train"][1]["context"][:300])  # show first 300 chars of context
print(dataset["train"][1]["question"])
print(dataset["train"][1]["answers"])
