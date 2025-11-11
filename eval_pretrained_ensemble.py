from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from datasets import load_dataset
import json
import argparse
import torch
from tqdm import tqdm


def ensemble_answer(question, context, qa_pipelines, weights):
    """
    Weighted confidence vote ensemble for question answering.
    Each model gets a vote weighted by its confidence score and assigned weight.
    """
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

    # Top 3 models based on F1 scores:
    # 1. aware-ai/xlmroberta-squadv2 (F1: 92.21)
    # 2. mrm8488/longformer-base-4096-finetuned-squadv2 (F1: 91.74)
    # 3. deepset/deberta-v3-large-squad2 (F1: 90.75)
    parser.add_argument(
        "--model_names", 
        type=str, 
        default="aware-ai/xlmroberta-squadv2,mrm8488/longformer-base-4096-finetuned-squadv2,deepset/deberta-v3-large-squad2",
        help="Comma-separated list of pre-trained model names from HuggingFace"
    )
    parser.add_argument(
        "--weights", 
        type=str,
        default="1.0,1.0,1.0",
        help="Comma-separated weights for each model (e.g., '1.0,0.95,0.90')"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="dev-v1.1.json",
        help="Test data file (SQuAD format JSON)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions_pretrained_ensemble.json",
        help="Output predictions file"
    )

    args = parser.parse_args()
    
    model_names = [m.strip() for m in args.model_names.split(",") if m.strip()]
    
    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(",")]
        if len(weights) != len(model_names):
            print(f"Warning: {len(weights)} weights provided but {len(model_names)} models specified. Using equal weights.")
            weights = [1.0] * len(model_names)
    else:
        weights = [1.0] * len(model_names)

    print(f"\n{'='*60}")
    print(f"Loading {len(model_names)} pre-trained models for ensemble:")
    print(f"{'='*60}")
    for i, (model_name, weight) in enumerate(zip(model_names, weights), 1):
        print(f"{i}. {model_name} (weight: {weight})")
    print(f"{'='*60}\n")

    # ===== 1. Load pre-trained models from HuggingFace =====
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA (GPU)' if device == 0 else 'CPU'}\n")
    
    tokenizers = []
    models = []
    qa_pipelines = []
    
    for i, model_name in tqdm(enumerate(model_names, 1), total=len(model_names), desc="Loading models"):
        print(f"[{i}/{len(model_names)}] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
        
        tokenizers.append(tokenizer)
        models.append(model)
        qa_pipelines.append(qa_pipeline)
        print(f"     ✓ Loaded successfully\n")

    # ===== 2. Load test data =====
    print(f"Loading test data from {args.test_file}...")
    dataset = load_dataset("json", data_files={"test": args.test_file}, field="data")
    print(f"✓ Loaded {len(dataset['test'])} articles\n")

    # ===== 3. Generate predictions =====
    print("Generating ensemble predictions...")
    predictions = {}
    total_questions = 0
    errors = 0

    # Count total questions for progress bar
    total_questions_count = sum(len(paragraph["qas"]) for article in dataset["test"] for paragraph in article["paragraphs"])
    
    for article in tqdm(dataset["test"], desc="Processing articles"):
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                qid = qa["id"]
                question = qa["question"]
                total_questions += 1
                
                try:
                    predictions[qid] = ensemble_answer(question, context, qa_pipelines, weights)
                except Exception as e:
                    print(f"Error for question {qid}: {e}")
                    predictions[qid] = ""
                    errors += 1

    print(f"\n✓ Generated predictions for {total_questions} questions")
    if errors > 0:
        print(f"  ⚠ {errors} errors encountered")
        
    # ===== 4. Save predictions =====
    with open(args.output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Saved predictions to {args.output_file}\n")
    print(f"{'='*60}")
    print(f"Next step: Run evaluation with:")
    print(f"  python evaluate-v2.0.py {args.test_file} {args.output_file}")
    print(f"{'='*60}")
