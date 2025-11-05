from datasets import load_dataset
from transformers import AutoTokenizer

# ------------- Load raw (nested) SQuAD JSON -------------
dataset = load_dataset(
    "json",
    data_files={"train": "train-v1.1.json", "test": "dev-v1.1.json"},
    field="data"
)
# ------------- Flatten to (id, context, question, answers) -------------
def flatten_squad(batch):
    # batch["paragraphs"] is a list whose elements are *lists of paragraph dicts*
    out = {"id": [], "context": [], "question": [], "answers": []}

    for paragraphs in batch["paragraphs"]:         # paragraphs: List[ {context, qas} ]
        for para in paragraphs:
            ctx = para["context"]
            for qa in para["qas"]:
                qid = qa["id"]
                q   = qa["question"]
                answers_list = qa.get("answers", [])

                # Take the first annotated answer (standard for SQuAD v1.1 fine-tuning)
                if answers_list:
                    first = answers_list[0]
                    # Normalize to dict-of-lists as HF expects
                    texts  = first["text"]
                    starts = first["answer_start"]
                    # Convert scalars to lists if needed
                    texts  = [texts]  if isinstance(texts, str) else texts
                    starts = [starts] if isinstance(starts, int) else starts
                else:
                    texts, starts = [], []

                out["id"].append(qid)
                out["context"].append(ctx)
                out["question"].append(q)
                out["answers"].append({"text": texts, "answer_start": starts})

    return out

# Remove the original nested columns ('title', 'paragraphs') after flattening
train_flat = dataset["train"].map(flatten_squad, remove_columns=["title", "paragraphs"], batched=True)
test_flat  = dataset["test"].map(flatten_squad, remove_columns=["title", "paragraphs"], batched=True)

train_val = train_flat.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val["train"]        # 90% for training
internal_val_dataset = train_val["test"]  # 10% for validation during training
test_flat.save_to_disk("data/flattened_squad_test")
# ------------- (Optional) sanity check a sample -------------
# print("SAMPLE:", dataset["train"][0])

# ------------- Tokenization for DeBERTa v3 -------------
model_checkpoint = "microsoft/deberta-v3-base"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128

def preprocess(batch):
    questions = [q.strip() for q in batch["question"]]
    encoded = tokenizer(
        questions,
        batch["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map     = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded.pop("offset_mapping")

    start_positions = []
    end_positions   = []

    for i, offsets in enumerate(offset_mapping):
        input_ids    = encoded["input_ids"][i]
        cls_index    = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = encoded.sequence_ids(i)
        sample_idx   = sample_map[i]

        # answers is dict-of-lists by construction
        ans = batch["answers"][sample_idx]
        if len(ans["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = ans["answer_start"][0]
        end_char   = start_char + len(ans["text"][0])

        # Find the start and end of the context in the tokenized sequence
        # sequence_ids == 1 marks context tokens (0=question, 1=context, None=special)
        # Find first/last index where sequence_ids == 1
        token_start = 0
        while sequence_ids[token_start] != 1:
            token_start += 1
        token_end = len(input_ids) - 1
        while sequence_ids[token_end] != 1:
            token_end -= 1

        # If answer is not fully inside this span, force to CLS
        if not (offsets[token_start][0] <= start_char and offsets[token_end][1] >= end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Move token_start to the right until we cross start_char
            while token_start < len(offsets) and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            # Move token_end to the left until we cross end_char
            while offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

    encoded["start_positions"] = start_positions
    encoded["end_positions"]   = end_positions
    return encoded
tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=["id", "context", "question", "answers"])
tokenized_val   = internal_val_dataset.map(preprocess, batched=True, remove_columns=["id", "context", "question", "answers"])

