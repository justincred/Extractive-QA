from datasets import load_dataset
from transformers import AutoTokenizer

# ------------- Load raw (nested) SQuAD JSON -------------
dataset = load_dataset(
    "json",
    data_files={"train": "train-v1.1.json", "validation": "dev-v1.1.json"},
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
dataset = dataset.map(
    flatten_squad,
    batched=True,
    remove_columns=dataset["train"].column_names
)

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

    sample_map = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded.pop("offset_mapping")
    answers = batch["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        ans = answers[sample_idx]
        input_ids = encoded["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = encoded.sequence_ids(i)

        start_char = ans["answer_start"][0]
        end_char = start_char + len(ans["text"][0])

        # Find the start and end of the context in the tokenized sequence
        # sequence_ids == 1 marks context tokens (0=question, 1=context, None=special)
        # Find first/last index where sequence_ids == 1
        token_idx = 0
        while sequence_ids[token_idx] != 1:
            token_idx += 1
        token_start = token_idx
        while sequence_ids[token_idx] == 1:
            token_idx += 1
        token_end = token_idx - 1

        # If answer is not fully inside this span, force to CLS
        if (offsets[token_start][0] > start_char or offsets[token_end][1] < end_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            token_idx = token_start
            # Move token_start to the right until we cross start_char
            while token_idx < token_end and offsets[token_idx][0] <= start_char:
                token_idx += 1
            start_positions.append(token_index - 1)

            token_idx = token_end
            # Move token_end to the left until we cross end_char
            while offsets[token_end][1] >= end_char and token_idx >= token_end:
                token_idx -= 1
            end_positions.append(token_idx + 1)

    encoded["start_positions"] = start_positions
    encoded["end_positions"] = end_positions
    return encoded

# Important: remove the flattened text columns here
tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["id", "context", "question", "answers"]
)

tokenized.save_to_disk("data/tokenized_squad_2")
print("Tokenized dataset saved to data/tokenized_squad_2")

