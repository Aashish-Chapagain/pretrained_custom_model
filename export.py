import argparse
import json
import os
import shutil
import sys
import torch

from config.settings import MODEL, TOKENIZER, TRAINING
from model import MiniLLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


STANDALONE_MODEL_PY = '''import torch
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    TransformerEncoder,
    TransformerEncoderLayer,
    init,
)


class MiniLLM(Module):
    """Standalone MiniLLM Transformer Architecture.

    Self-contained decoder-style transformer with causal masking.
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 256,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.pos_embedding = Embedding(max_seq_len, embedding_dim)
        self.register_buffer(
            "causal_mask",
            self._generate_causal_mask(max_seq_len),
            persistent=False,
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(embedding_dim),
            enable_nested_tensor=False,
        )
        self.output_layer = Linear(embedding_dim, vocab_size, bias=False)
        init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def _generate_causal_mask(self, max_len: int) -> torch.Tensor:
        mask = torch.full((max_len, max_len), float("-inf"))
        return torch.triu(mask, diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        attn_mask = self.causal_mask[:seq_len, :seq_len]
        output = self.transformer(x, mask=attn_mask)
        return self.output_layer(output)
'''

STANDALONE_RUNTIME_PY = '''import json
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
                user_text = input("\\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\\nExiting chat. Goodbye!")
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

            user_turn_text = f"\\nUser: {user_text}\\nAssistant: "
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
'''

INIT_PY = '''from .model import MiniLLM
from .minillm import MiniLLMRuntime, load_model

__all__ = ["MiniLLM", "MiniLLMRuntime", "load_model"]
'''

EXAMPLE_USAGE_PY = '''"""Example showing how to load and use MiniLLM in any external project."""
from minillm import MiniLLMRuntime, load_model

# 1. Load the model from this directory
llm = load_model(".")

# 2. Single generation
prompt = "The solar system"
output = llm.generate(prompt, max_new_tokens=60, temperature=0.35)
print("Generated:")
print(output)

# 3. Or start interactive chat:
# llm.chat()
'''

EXPORT_README_MD = '''# Exported MiniLLM Package

This directory contains the trained **MiniLLM** model weights, architecture, and tokenizer bundled as a self-contained runtime package.

## Quick Start in Any Python Project

Copy this entire folder (or add it to your Python path), then:

```python
from minillm import load_model

# Load model, weights, and tokenizer automatically
llm = load_model("path/to/this/folder")

# Generate text:
print(llm.generate("Artificial intelligence is", max_new_tokens=80))

# Or launch interactive chat:
llm.chat()
```

## Contents
- `weights.pth`: Trained state dictionary weights.
- `full_model.pth`: Full PyTorch serialized model object.
- `tokenizer.model`: SentencePiece BPE tokenizer.
- `config.json`: Model architecture hyperparameters.
- `model.py`: Standalone PyTorch architecture definition of `MiniLLM`.
- `minillm.py`: Single-class runtime with generation, sampling, and chat.
'''


def export_model(
    output_dir: str = "exported_minillm",
    weights_path: str = None,
    tokenizer_path: str = None,
) -> str:
    """Export the trained model, architecture, config, and tokenizer into a portable directory."""
    if weights_path is None:
        weights_path = TRAINING.get("final_model_path", "final_model.pth")
        if not os.path.exists(weights_path):
            weights_path = TRAINING.get("checkpoint_path", "minillm_checkpoint.pth")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights file not found at '{weights_path}'.\n"
            "Please ensure you have trained the model or specified a valid weights path."
        )

    if tokenizer_path is None:
        tokenizer_path = f"{TOKENIZER['model_prefix']}.model"

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found at '{tokenizer_path}'.\n"
            "Please train or provide the tokenizer model."
        )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting MiniLLM package to: '{os.path.abspath(output_dir)}'...")

    # 1. Write config.json
    config_data = {
        "model_type": "minillm",
        "vocab_size": MODEL["vocab_size"],
        "embedding_dim": MODEL["embedding_dim"],
        "num_heads": MODEL["num_heads"],
        "num_layers": MODEL["num_layers"],
        "max_seq_len": MODEL["max_seq_len"],
        "ffn_dim": MODEL["ffn_dim"],
        "dropout": MODEL.get("dropout", 0.1),
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print("  [OK] Saved config.json")

    # 2. Copy Tokenizer
    dest_tokenizer = os.path.join(output_dir, "tokenizer.model")
    shutil.copy2(tokenizer_path, dest_tokenizer)
    print("  [OK] Copied tokenizer.model")

    # 3. Load & Save weights (state dict)
    dest_weights = os.path.join(output_dir, "weights.pth")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError(f"Unrecognized checkpoint format in {weights_path}")

    torch.save(state_dict, dest_weights)
    print("  [OK] Saved weights.pth (state dictionary)")

    # 4. Save full PyTorch serialized model object
    model = MiniLLM(
        vocab_size=MODEL["vocab_size"],
        embedding_dim=MODEL["embedding_dim"],
        num_heads=MODEL["num_heads"],
        num_layers=MODEL["num_layers"],
        max_seq_len=MODEL["max_seq_len"],
        ffn_dim=MODEL["ffn_dim"],
        dropout=MODEL.get("dropout", 0.1),
    )
    model.load_state_dict(state_dict)
    model.eval()
    dest_full = os.path.join(output_dir, "full_model.pth")
    torch.save(model, dest_full)
    print("  [OK] Saved full_model.pth (serialized PyTorch model object)")

    # 5. Write Python runtime files
    with open(os.path.join(output_dir, "model.py"), "w", encoding="utf-8") as f:
        f.write(STANDALONE_MODEL_PY)
    print("  [OK] Wrote standalone model.py")

    with open(os.path.join(output_dir, "minillm.py"), "w", encoding="utf-8") as f:
        f.write(STANDALONE_RUNTIME_PY)
    print("  [OK] Wrote standalone minillm.py")

    with open(os.path.join(output_dir, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(INIT_PY)
    print("  [OK] Wrote __init__.py")

    with open(os.path.join(output_dir, "example_usage.py"), "w", encoding="utf-8") as f:
        f.write(EXAMPLE_USAGE_PY)
    print("  [OK] Wrote example_usage.py")

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(EXPORT_README_MD)
    print("  [OK] Wrote README.md")

    print("\nExport complete! The model package is ready in:")
    print(f"  --> {os.path.abspath(output_dir)}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export MiniLLM architecture, weights, and tokenizer into a standalone package."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="exported_minillm",
        help="Destination directory for exported package.",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to trained weights (.pth file).",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=None,
        help="Path to SentencePiece tokenizer model.",
    )
    args = parser.parse_args()
    export_model(
        output_dir=args.output,
        weights_path=args.weights,
        tokenizer_path=args.tokenizer,
    )


if __name__ == "__main__":
    main()
