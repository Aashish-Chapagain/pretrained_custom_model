import argparse
import os
import sys
import time
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import sentencepiece as spm
import torch
import torch.nn.functional as F

from config.settings import MODEL, SFT, TOKENIZER, TRAINING
from model import MiniLLM
from train import resolve_device


class MiniLLMInference:
    """High-level inference engine for fine-tuned and pretrained MiniLLM."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or resolve_device()
        print(f"Using compute device: {self.device}")

        # 1. Resolve Tokenizer
        if tokenizer_path is None:
            tokenizer_path = f"{TOKENIZER['model_prefix']}.model"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer file '{tokenizer_path}' not found.\n"
                "Please run 'python tokenizer.py' first."
            )
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.eos_id = self.tokenizer.eos_id() if self.tokenizer.eos_id() >= 0 else 3
        self.pad_id = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else 0

        # 2. Resolve Weights
        if weights_path is None:
            sft_path = SFT.get("finetuned_model_path", "finetuned_model.pth")
            final_path = TRAINING.get("final_model_path", "final_model.pth")
            ckpt_path = TRAINING.get("checkpoint_path", "minillm_checkpoint.pth")

            if os.path.exists(sft_path):
                weights_path = sft_path
                self.model_type = "Instruction Fine-Tuned (SFT)"
            elif os.path.exists(final_path):
                weights_path = final_path
                self.model_type = "Pretrained Base Model"
            elif os.path.exists(ckpt_path):
                weights_path = ckpt_path
                self.model_type = "Training Checkpoint"
            else:
                raise FileNotFoundError(
                    "No model weights found. Expected 'finetuned_model.pth' or 'final_model.pth'."
                )
        else:
            self.model_type = "Custom Weights"

        # 3. Instantiate Architecture and Load Weights
        self.model = MiniLLM(
            vocab_size=MODEL["vocab_size"],
            embedding_dim=MODEL["embedding_dim"],
            num_heads=MODEL["num_heads"],
            num_layers=MODEL["num_layers"],
            max_seq_len=MODEL["max_seq_len"],
            ffn_dim=MODEL["ffn_dim"],
            dropout=0.0,
        )

        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.weights_path = weights_path
        print(f"Loaded {self.model_type} from: '{weights_path}'")

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.4,
        top_k: int = 20,
        top_p: float = 0.85,
    ) -> int:
        logits = logits.cpu().float()
        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        logits = logits / max(temperature, 1e-5)

        if 0 < top_k < logits.size(-1):
            threshold = torch.topk(logits, top_k)[0][-1]
            logits[logits < threshold] = float("-inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            filtered = sorted_logits.masked_fill(remove_mask, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, sorted_indices, filtered)

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return int(torch.argmax(logits).item())

        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def ask(
        self,
        question: str,
        max_new_tokens: int = 100,
        temperature: float = 0.4,
        top_k: int = 20,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
        stream: bool = True,
    ) -> str:
        """Send an instruction / query to the model and receive a direct answer."""
        formatted_prompt = f"User: {question.strip()}\nAssistant: "
        tokens = self.tokenizer.encode(formatted_prompt)

        generated_tokens = []
        prev_text = self.tokenizer.decode(tokens)

        if stream:
            print("Assistant: ", end="", flush=True)

        start_time = time.perf_counter()

        for _ in range(max_new_tokens):
            context = tokens[-self.model.max_seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=self.device)
            logits = self.model(x)[0, -1]

            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_tokens:
                counts = {}
                for t in generated_tokens:
                    counts[t] = counts.get(t, 0) + 1
                for t, c in counts.items():
                    if logits[t] > 0:
                        logits[t] /= repetition_penalty**c
                    else:
                        logits[t] *= repetition_penalty**c

            next_id = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Check stop conditions
            if next_id == self.eos_id:
                break

            tokens.append(next_id)
            generated_tokens.append(next_id)

            if stream:
                full_text = self.tokenizer.decode(tokens)
                delta = full_text[len(prev_text) :]
                print(delta, end="", flush=True)
                prev_text = full_text

        elapsed = time.perf_counter() - start_time
        response_text = self.tokenizer.decode(generated_tokens).strip()

        # Clean up any trailing 'User:' or '</s>' artifact
        if response_text.endswith("User:"):
            response_text = response_text[:-5].strip()

        if stream:
            print()
            tok_per_sec = len(generated_tokens) / max(elapsed, 1e-4)
            print(f"[{len(generated_tokens)} tokens generated in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]")

        return response_text

    def chat(
        self,
        max_new_tokens: int = 100,
        temperature: float = 0.4,
        top_k: int = 20,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
    ) -> None:
        """Run an interactive multi-turn terminal conversation with memory."""
        print("=" * 65)
        print(f" MiniLLM Interactive Chat ({self.model_type})")
        print(" Commands: 'clear' to reset conversation, 'exit' to quit.")
        print("=" * 65 + "\n")

        history_tokens: list[int] = []
        max_context = self.model.max_seq_len - max_new_tokens

        while True:
            try:
                user_msg = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat. Goodbye!")
                break

            if not user_msg:
                continue

            lowered = user_msg.lower()
            if lowered in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break

            if lowered in ("clear", "reset"):
                history_tokens.clear()
                print("\n[Conversation memory cleared]\n")
                continue

            # Format turn
            turn_str = f"User: {user_msg}\nAssistant: " if not history_tokens else f"\nUser: {user_msg}\nAssistant: "
            turn_tokens = self.tokenizer.encode(turn_str)

            # Context window management
            if len(history_tokens) + len(turn_tokens) > max_context:
                keep = max_context - len(turn_tokens)
                history_tokens = history_tokens[-keep:] if keep > 0 else []

            history_tokens.extend(turn_tokens)

            print("Assistant: ", end="", flush=True)
            prev_text = self.tokenizer.decode(history_tokens)
            generated_turn = []

            for _ in range(max_new_tokens):
                context = history_tokens[-self.model.max_seq_len :]
                x = torch.tensor([context], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(x)[0, -1]

                if repetition_penalty != 1.0 and generated_turn:
                    counts = {}
                    for t in generated_turn:
                        counts[t] = counts.get(t, 0) + 1
                    for t, c in counts.items():
                        if logits[t] > 0:
                            logits[t] /= repetition_penalty**c
                        else:
                            logits[t] *= repetition_penalty**c

                next_id = self._sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                if next_id == self.eos_id:
                    break

                history_tokens.append(next_id)
                generated_turn.append(next_id)

                full_text = self.tokenizer.decode(history_tokens)
                delta = full_text[len(prev_text) :]
                print(delta, end="", flush=True)
                prev_text = full_text

            print()


def main():
    parser = argparse.ArgumentParser(
        description="Inference CLI for fine-tuned and pretrained MiniLLM."
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Single prompt/question to ask the model.",
    )
    parser.add_argument(
        "--chat",
        "-c",
        action="store_true",
        help="Launch continuous interactive chat session.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to specific weights (.pth file). Defaults to finetuned_model.pth.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate (default: 100).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (default: 0.4).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling cutoff (default: 20).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-P nucleus cutoff (default: 0.85).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty factor (default: 1.2).",
    )
    args = parser.parse_args()

    engine = MiniLLMInference(weights_path=args.weights)

    if args.prompt:
        engine.ask(
            question=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stream=True,
        )
    else:
        # Default to interactive chat if no prompt provided
        engine.chat(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
