import argparse
import os

import sentencepiece as spm
import torch

from config.settings import GENERATE, TOKENIZER, TRAINING
from model import MiniLLM
from train import resolve_device


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    matches = (preds == targets).float()
    return float(matches.mean().item())


def evaluate_sequence_accuracy(
    model: MiniLLM,
    prompt: str,
    target_text: str,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
) -> float:
    prompt_tokens = tokenizer.encode(prompt)
    target_tokens = tokenizer.encode(target_text)
    if not target_tokens:
        raise ValueError("target_text must contain at least one token")

    total_predictions = 0
    correct_predictions = 0
    context = prompt_tokens[-model.max_seq_len :]

    for idx, target_id in enumerate(target_tokens):
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        pred = int(torch.argmax(logits).item())

        total_predictions += 1
        if pred == target_id:
            correct_predictions += 1

        next_token = target_id
        context = (context + [next_token])[-model.max_seq_len :]

    if total_predictions == 0:
        return 0.0
    return correct_predictions / total_predictions


def load_model(device: torch.device) -> MiniLLM:
    model = MiniLLM()
    final_path = TRAINING["final_model_path"]
    checkpoint_path = TRAINING["checkpoint_path"]

    if os.path.exists(final_path):
        state = torch.load(final_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded {final_path}")
    elif os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded {checkpoint_path}")
    else:
        raise FileNotFoundError(
            f"No weights found. Train first, expected {final_path} or {checkpoint_path}."
        )

    model.to(device)
    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: MiniLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_id()
    max_seq_len = model.max_seq_len

    for _ in range(max_new_tokens):
        context = tokens[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        next_id = sample_next_token(logits, temperature)
        tokens.append(next_id)
        if eos_id >= 0 and next_id == eos_id:
            break

    return tokenizer.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with MiniLLM")
    parser.add_argument("--prompt", type=str, default=GENERATE["prompt"])
    parser.add_argument("--max-new-tokens", type=int, default=GENERATE["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=GENERATE["temperature"])
    parser.add_argument("--eval-text", type=str, default="")
    parser.add_argument("--eval-prompt", type=str, default="")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    tokenizer = spm.SentencePieceProcessor(
        model_file=f"{TOKENIZER['model_prefix']}.model"
    )
    model = load_model(device)

    if args.eval_text:
        eval_prompt = args.eval_prompt or args.prompt
        accuracy = evaluate_sequence_accuracy(
            model,
            prompt=eval_prompt,
            target_text=args.eval_text,
            tokenizer=tokenizer,
            device=device,
        )
        print(f"Sequence accuracy: {accuracy:.4f}")

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
