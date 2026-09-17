import json
import os
import sys
from typing import Optional
import sentencepiece as spm
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from .model import MiniLLM
except ImportError:
    from model import MiniLLM


def resolve_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MiniLLMRuntime:
    """Complete, self-contained inference runtime for MiniLLM."""

    def __init__(
        self,
        model: MiniLLM,
        tokenizer: spm.SentencePieceProcessor,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or resolve_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = config.get("max_seq_len", 256)
        self.eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else 3

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str = ".",
        device: Optional[torch.device] = None,
    ) -> "MiniLLMRuntime":
        """Load architecture, weights, and tokenizer from an exported model folder."""
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "weights.pth")
        if not os.path.exists(weights_path):
            alt_weights = os.path.join(model_dir, "final_model.pth")
            if os.path.exists(alt_weights):
                weights_path = alt_weights
            else:
                raise FileNotFoundError(f"Weights file not found in '{model_dir}'.")

        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at '{tokenizer_path}'.")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "vocab_size": 5000,
                "embedding_dim": 256,
                "num_heads": 8,
                "num_layers": 6,
                "max_seq_len": 256,
                "ffn_dim": 1024,
                "dropout": 0.1,
            }

        target_device = device or resolve_device()
        model = MiniLLM(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"],
            ffn_dim=config["ffn_dim"],
            dropout=config.get("dropout", 0.1),
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        return cls(model=model, tokenizer=tokenizer, config=config, device=target_device)

    def sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.35,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> int:
        logits = logits.cpu().float()
        if temperature <= 0.0:
            return int(torch.argmax(logits).item())

        logits = logits / max(temperature, 1e-5)

        if top_k > 0 and top_k < logits.size(-1):
            threshold = torch.topk(logits, top_k)[0][-1]
            logits[logits < threshold] = float("-inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))
            logits = torch.empty_like(logits).scatter_(0, sorted_indices, filtered_sorted)

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return int(torch.argmax(logits).item())

        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.35,
        top_k: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        stream: bool = False,
    ) -> str:
        """Autoregressively generate text given a prompt string."""
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [self.tokenizer.bos_id() if hasattr(self.tokenizer, "bos_id") else 2]

        generated_tokens = []
        prev_text = self.tokenizer.decode(tokens)
        if stream:
            print(prev_text, end="", flush=True)

        for _ in range(max_new_tokens):
            context = tokens[-self.max_seq_len :]
            x = torch.tensor([context], dtype=torch.long, device=self.device)
            logits = self.model(x)[0, -1]

            if repetition_penalty != 1.0 and generated_tokens:
                counts = {}
                for tid in generated_tokens:
                    counts[tid] = counts.get(tid, 0) + 1
                for tid, count in counts.items():
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty**count
                    else:
                        logits[tid] *= repetition_penalty**count

            next_id = self.sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens.append(next_id)
            generated_tokens.append(next_id)

            if stream:
                full_text = self.tokenizer.decode(tokens)
                delta = full_text[len(prev_text) :]
                print(delta, end="", flush=True)
                prev_text = full_text

            if self.eos_id >= 0 and next_id == self.eos_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(tokens)

    def chat(
        self,
        temperature: float = 0.35,
        top_k: int = 10,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> None:
        """Start an interactive multi-turn terminal chat session."""
        print("=" * 60)
        print("MiniLLM Interactive Chat Session")
        print("Commands: 'clear' / 'reset' to clear history, 'exit' / 'quit' to exit.")
        print("=" * 60)

        history_tokens: list[int] = []

        while True:
            try:
                user_text = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat. Goodbye!")
                break

            if not user_text:
                continue

            lowered = user_text.lower()
            if lowered in ("exit", "quit"):
                print("Exiting chat. Goodbye!")
                break

            if lowered in ("clear", "reset"):
                history_tokens.clear()
                print("Conversation history cleared.")
                continue

            user_turn_text = f"\nUser: {user_text}\nAssistant: "
            user_turn_tokens = self.tokenizer.encode(user_turn_text)

            max_history = self.max_seq_len - 80
            if len(history_tokens) + len(user_turn_tokens) > max_history:
                keep_len = max_history - len(user_turn_tokens)
                if keep_len > 0:
                    history_tokens = history_tokens[-keep_len:]
                else:
                    history_tokens.clear()

            history_tokens.extend(user_turn_tokens)
            prev_text = self.tokenizer.decode(history_tokens)

            print("Assistant: ", end="", flush=True)
            generated_turn = []

            for _ in range(80):
                context = history_tokens[-self.max_seq_len :]
                x = torch.tensor([context], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(x)[0, -1]

                if repetition_penalty != 1.0 and generated_turn:
                    counts = {}
                    for tid in generated_turn:
                        counts[tid] = counts.get(tid, 0) + 1
                    for tid, count in counts.items():
                        if logits[tid] > 0:
                            logits[tid] /= repetition_penalty**count
                        else:
                            logits[tid] *= repetition_penalty**count

                next_id = self.sample_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                history_tokens.append(next_id)
                generated_turn.append(next_id)

                full_text = self.tokenizer.decode(history_tokens)
                delta = full_text[len(prev_text) :]
                print(delta, end="", flush=True)
                prev_text = full_text

                if self.eos_id >= 0 and next_id == self.eos_id:
                    break

            print()


def load_model(model_dir: str = ".", device: Optional[torch.device] = None) -> MiniLLMRuntime:
    """Convenience alias to load MiniLLMRuntime from an export directory."""
    return MiniLLMRuntime.from_pretrained(model_dir=model_dir, device=device)
