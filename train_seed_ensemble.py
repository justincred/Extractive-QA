import os
from datasets import load_from_disk
from transformers import (
    set_seed,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import argparse
import torch

print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def load_tokenized_splits(data_dir):
    train_ds = load_from_disk(f"{data_dir}/tokenized_squad_train")
    val_ds = load_from_disk(f"{data_dir}/tokenized_squad_internal_val")

    return train_ds, val_ds


def train_single_model(model_name="microsoft/deberta-v3-base", data_dir="data", output_dir="models/some_model", lr=3e-5, epochs=3, batch_size=8, weight_decay=0.01, seed=42):
    
    print(f"Training model: {model_name} with seed {seed}")

    set_seed(seed)

    safe_name = model_name.replace("/", "_")
    train_ds, val_ds = load_tokenized_splits(os.path.join(data_dir, safe_name))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        fp16=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved to:", output_dir)


def train_multiple_models(model_name, seeds, data_dir="data", output_dir="models", lr=3e-5, epochs=3, batch_size=8, weight_decay=0.01):
    os.makedirs(output_dir, exist_ok=True)

    model_dirs = []
    safe_name = model_name.replace("/", "_")

    for seed in seeds:
        model_output_dir = os.path.join(output_dir, f"{safe_name}_seed{seed}")

        train_single_model(
            model_name=model_name,
            data_dir=data_dir,
            output_dir=model_output_dir,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            seed=seed,
        )

        model_dirs.append(model_output_dir)

    print("All seeded models trained. Ensemble seed models:")
    for d in model_dirs:
        print(" -", d)
    
    return model_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--seeds", type=str, default="42,43,44")

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    train_multiple_models(
        model_name=args.model_name,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

