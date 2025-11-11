from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import json
import argparse


def ensemble_answer(question, context, qa_pipelines, weights):
    results = [pipe(question=question, context=context) for pipe in qa_pipelines]

    # Weighted confidence vote ensemble

    candidates = {}

    for r, w in zip(results, weights):
        ans = " ".join(r["answer"].strip().lower().split())
        conf = float(r.get("score", 1.0))

        if ans == "":
            continue
        
        if ans not in candidates:
            candidates[ans] = 0.0
        candidates[ans] += w * conf

    if not candidates:
        return ""

    return max(candidates.items(), key=lambda x: x[1])[0]
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_names", type=str, default="microsoft/deberta-v3-base,bert-base-uncased,roberta-base")
    parser.add_argument("--weights", type=str)

    args = parser.parse_args()
    model_paths = [f"models/{m.strip().replace('/', '_')}" for m in args.model_names.split(",") if m.strip()]

    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(",")]
        if len(weights) != len(model_paths):
            weights = [1.0] * len(model_paths)
    else:
        weights = [1.0] * len(model_paths)


    # ===== 1. Load fine-tuned model =====
    tokenizers = [AutoTokenizer.from_pretrained(model_path, use_fast=True) for model_path in model_paths]
    models = [AutoModelForQuestionAnswering.from_pretrained(model_path) for model_path in model_paths]
    qa_pipelines = [pipeline("question-answering", model=model, tokenizer=tokenizer, device=0) for model, tokenizer in zip(models, tokenizers)]


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
                    predictions[qid] = ensemble_answer(question, context, qa_pipelines, weights)
                except Exception as e:
                    print(f"Error for id {qid}: {e}")
                    predictions[qid] = ""

    
        
    # ===== 4. Save predictions =====
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)
    print(f" Saved predictions for {len(predictions)} questions to predictions.json")