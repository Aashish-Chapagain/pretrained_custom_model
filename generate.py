import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    loaded = False
    for path, is_checkpoint in [(final_path, False), (checkpoint_path, True)]:
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = state["model_state_dict"] if is_checkpoint else state
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded {path}")
                loaded = True
                break
            except RuntimeError as exc:
                print(
                    f"Notice: Existing weights in '{path}' are from an earlier model version with different dimensions: {exc}"
                )

    if not loaded:
        raise RuntimeError(
            "No compatible weights found for the current model architecture (e.g. sequence length changed).\n"
            "Please run 'python train.py' to train the model on the new dataset."
        )

    model.to(device)
    model.eval()
    return model


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: list[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    if penalty == 1.0 or not generated_tokens:
        return logits
    logits = logits.clone()
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = GENERATE.get("temperature", 0.35),
    top_k: int = GENERATE.get("top_k", 10),
    top_p: float = GENERATE.get("top_p", 0.9),
) -> int:
    logits = logits.cpu().float()
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    # Top-K filtering
    if 0 < top_k < logits.size(-1):
        top_k_values, _ = torch.topk(logits, top_k)
        min_value = top_k_values[-1]
        logits = torch.where(
            logits < min_value,
            torch.tensor(float("-inf")),
            logits,
        )

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or (probs <= 0).all():
        return int(torch.argmax(logits).item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: MiniLLM,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int = GENERATE["max_new_tokens"],
    temperature: float = GENERATE["temperature"],
    device: torch.device = None,
    top_k: int = GENERATE.get("top_k", 10),
    top_p: float = GENERATE.get("top_p", 0.9),
    repetition_penalty: float = GENERATE.get("repetition_penalty", 1.2),
    stream: bool = False,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt)
    eos_id = tokenizer.eos_id()
    max_seq_len = model.max_seq_len
    generated_tokens = []

    prev_text = prompt
    if stream:
        print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        context = tokens[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1]

        if repetition_penalty != 1.0 and generated_tokens:
            logits = apply_repetition_penalty(
                logits, generated_tokens, repetition_penalty
            )

        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens.append(next_id)
        generated_tokens.append(next_id)

        if stream:
            full_text = tokenizer.decode(tokens)
            delta = full_text[len(prev_text) :]
            print(delta, end="", flush=True)
            prev_text = full_text

        if eos_id >= 0 and next_id == eos_id:
            break

    if stream:
        print()

    return tokenizer.decode(tokens)


def chat_loop(
    model: MiniLLM,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    max_new_tokens: int = GENERATE.get("max_new_tokens", 80),
    temperature: float = GENERATE.get("temperature", 0.35),
    top_k: int = GENERATE.get("top_k", 10),
    top_p: float = GENERATE.get("top_p", 0.9),
    repetition_penalty: float = GENERATE.get("repetition_penalty", 1.2),
) -> None:
    """Runs a continuous interactive multi-turn chat session with rolling context."""
    print("\n" + "=" * 65)
    print(" MiniLLM Continuous Interactive Chat")
    print(" - Type your message and press Enter to chat.")
    print(" - Type 'clear' or 'reset' to clear conversation memory.")
    print(" - Type 'exit' or 'quit' to leave the chat.")
    print("=" * 65 + "\n")

    history_tokens: list[int] = []
    eos_id = tokenizer.eos_id()
    max_context = model.max_seq_len - max_new_tokens

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break

        if user_input.lower() in ["clear", "reset"]:
            history_tokens = []
            print("\n[Conversation context reset to start]\n")
            continue

        # Append turn to context
        if not history_tokens:
            prompt_turn = f"{user_input}\n"
        else:
            prompt_turn = f"\n{user_input}\n"

        history_tokens.extend(tokenizer.encode(prompt_turn))

        # Sliding context window to always fit within max_seq_len
        if len(history_tokens) > max_context:
            history_tokens = history_tokens[-max_context:]

        print("MiniLLM: ", end="", flush=True)
        prev_text = tokenizer.decode(history_tokens)
        generated_turn: list[int] = []

        for _ in range(max_new_tokens):
            context = history_tokens[-model.max_seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(x)[0, -1]

            if repetition_penalty != 1.0 and generated_turn:
                logits = apply_repetition_penalty(
                    logits, generated_turn, repetition_penalty
                )

            next_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            history_tokens.append(next_id)
            generated_turn.append(next_id)

            full_text = tokenizer.decode(history_tokens)
            delta = full_text[len(prev_text) :]
            print(delta, end="", flush=True)
            prev_text = full_text

            if eos_id >= 0 and next_id == eos_id:
                break

        print("\n")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate text with MiniLLM on Cosmopedia"
    )
    parser.add_argument("--prompt", type=str, default=GENERATE["prompt"])
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch continuous interactive chat mode",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=GENERATE["max_new_tokens"]
    )
    parser.add_argument(
        "--temperature", type=float, default=GENERATE.get("temperature", 0.35)
    )
    parser.add_argument("--top-k", type=int, default=GENERATE.get("top_k", 10))
    parser.add_argument("--top-p", type=float, default=GENERATE.get("top_p", 0.9))
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=GENERATE.get("repetition_penalty", 1.2),
    )
    parser.add_argument(
        "--stream", action="store_true", help="Stream generated tokens in real-time"
    )
    parser.add_argument("--eval-text", type=str, default="")
    parser.add_argument("--eval-prompt", type=str, default="")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    tokenizer_file = f"{TOKENIZER['model_prefix']}.model"
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(
            f"Tokenizer model '{tokenizer_file}' not found.\n"
            "Please run 'python tokenizer.py' first."
        )

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)
    model = load_model(device)

    if args.chat:
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        return

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

    if not args.stream:
        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=False,
        )
        print(text)
    else:
        generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=True,
        )


if __name__ == "__main__":
    main()
