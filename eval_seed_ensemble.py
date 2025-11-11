from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import json
import torch
import argparse


def ensemble_answer(question, context, tokenizer, models, weights, device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoded = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop("offset_mapping")[0]
    sequence_ids = encoded.sequence_ids(0)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    start_sum = None
    end_sum = None

    # Weighted sum of logits from each model
    for model, w in zip(models, weights):
        outputs = model(**encoded)
        s = outputs.start_logits[0] * w
        e = outputs.end_logits[0] * w
        if start_sum is None:
            start_sum, end_sum = s, e
        else:
            start_sum += s
            end_sum += e

    # Consider only context tokens
    context_indices = [i for i, s in enumerate(sequence_ids) if s == 1]
    if not context_indices:
        return ""

    best_start, best_end, best_score = 0, 0, float("-inf")

    for i in context_indices:
        for j in context_indices:
            if j < i:
                continue
            if j - i + 1 > 30:
                break
            score = start_sum[i].item() + end_sum[j].item()
            if score > best_score:
                best_score, best_start, best_end = score, i, j

    start_char = int(offset_mapping[best_start][0])
    end_char = int(offset_mapping[best_end][1])
    return context[start_char:end_char].strip()
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--weights", type=str)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    safe_name = args.model_name.replace("/", "_")
    model_paths = [f"models/{safe_name}_seed{s}" for s in args.seeds.split(",") if s.strip()]

    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(",")]
        if len(weights) != len(model_paths):
            weights = [1.0] * len(model_paths)
    else:
        weights = [1.0] * len(model_paths)


    # ===== 1. Load fine-tuned model =====
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], use_fast=True)
    models = []
    for path in model_paths:
        model = AutoModelForQuestionAnswering.from_pretrained(path).to(device)
        model.eval()
        models.append(model)

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
                    predictions[qid] = ensemble_answer(question, context, tokenizer, models, weights, device)
                except Exception as e:
                    print(f"Error for id {qid}: {e}")
                    predictions[qid] = ""

    
        
    # ===== 4. Save predictions =====
    with open("predictions.json", "w") as f:
        json.dump(predictions, f)
    print(f" Saved predictions for {len(predictions)} questions to predictions.json")