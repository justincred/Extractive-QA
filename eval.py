from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import json

# ===== 1. Load fine-tuned model =====
model_path = "models/deberta_squad"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

# ===== 2. Load test data =====
dataset = load_dataset("json", data_files={"test": "dev-v1.1.json"}, field="data")

# ===== 3. Generate predictions =====
predictions = {}

for article in dataset["test"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            qid = qa["id"]
            question = qa["question"]
            try:
                result = qa_pipeline(question=question, context=context)
                predictions[qid] = result["answer"]
            except Exception as e:
                print(f"Error for id {qid}: {e}")
                predictions[qid] = ""

# ===== 4. Save predictions =====
with open("predictions.json", "w") as f:
    json.dump(predictions, f)
print(f" Saved predictions for {len(predictions)} questions to predictions.json")
