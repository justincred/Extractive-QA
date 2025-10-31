import os
from datasets import load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from evaluate import load as load_metric
import torch
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
def train(model_name="microsoft/deberta-v3-base", data_dir="data", output_dir="../models/deberta_squad"):
    tokenized = load_from_disk(f"{data_dir}/tokenized_squad")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir="../logs",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(" Model saved to:", output_dir)

if __name__ == "__main__":
    train()

