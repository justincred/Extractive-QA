import os
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import argparse
from evaluate import load as load_metric
import torch

print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def train(model_name="microsoft/deberta-v3-base", data_dir="data", output_dir="models/deberta_squad", lr=3e-5, epochs=3, batch_size=8, weight_decay=0.01):
    tokenized_train = load_from_disk(f"{data_dir}/tokenized_squad_train")
    tokenized_val   = load_from_disk(f"{data_dir}/tokenized_squad_internal_val")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    args = parser.parse_args()    
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

