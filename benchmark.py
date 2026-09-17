import math

import torch
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from model import MiniLLM
from config.settings import MODEL, TOKENIZER, TRAINING


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER['model_prefix']}.model")


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

model = MiniLLM()

state_dict = torch.load(TRAINING["final_model_path"], map_location="cpu")

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print("Model loaded.")


# --------------------------------------------------
# Tokenize
# --------------------------------------------------


def encode(text):
    return sp.encode(text, out_type=int)


# --------------------------------------------------
# Score one continuation
# --------------------------------------------------


@torch.no_grad()
def score_continuation(context, continuation):
    """
    Calculates the average log probability of the continuation
    given the context.
    """

    context_tokens = encode(context)
    continuation_tokens = encode(continuation)

    if len(continuation_tokens) == 0:
        return float("-inf")

    # We need at least one context token
    if len(context_tokens) == 0:
        context_tokens = [TOKENIZER["bos_id"]]

    # Keep only enough context to fit max sequence length
    max_context = MODEL["max_seq_len"] - len(continuation_tokens)

    if max_context <= 0:
        return float("-inf")

    context_tokens = context_tokens[-max_context:]

    tokens = context_tokens + continuation_tokens

    # Truncate if necessary
    tokens = tokens[: MODEL["max_seq_len"]]

    input_ids = torch.tensor(
        [tokens[:-1]],
        dtype=torch.long,
        device=device,
    )

    target_ids = torch.tensor(
        tokens[1:],
        dtype=torch.long,
        device=device,
    )

    logits = model(input_ids)

    log_probs = torch.log_softmax(logits, dim=-1)

    # Number of context tokens.
    # We only want likelihood of continuation.
    continuation_start = len(context_tokens) - 1

    continuation_log_probs = []

    for i in range(continuation_start, len(target_ids)):
        token_id = target_ids[i]

        token_log_prob = log_probs[0, i, token_id]

        continuation_log_probs.append(token_log_prob.item())

    if not continuation_log_probs:
        return float("-inf")

    # Average rather than total log probability.
    # This prevents shorter answers from being unfairly favored.
    return sum(continuation_log_probs) / len(continuation_log_probs)


# --------------------------------------------------
# HellaSwag evaluation
# --------------------------------------------------

print("Loading HellaSwag...")

dataset = load_dataset(
    "Rowan/hellaswag",
    split="validation",
)

print(f"Examples: {len(dataset)}")


correct = 0
total = 0


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

for example in tqdm(dataset, desc="Evaluating HellaSwag"):

    context = example["ctx"]
    endings = example["endings"]
    correct_answer = int(example["label"])

    scores = []

    for ending in endings:

        score = score_continuation(
            context,
            ending,
        )

        scores.append(score)

    prediction = max(
        range(len(scores)),
        key=lambda i: scores[i],
    )

    if prediction == correct_answer:
        correct += 1

    total += 1


# --------------------------------------------------
# Results
# --------------------------------------------------

accuracy = 100.0 * correct / total

print()
print("=" * 50)
print("HellaSwag Benchmark")
print("=" * 50)
print(f"Correct:  {correct}")
print(f"Total:    {total}")
print(f"Accuracy: {accuracy:.2f}%")
print("=" * 50)
