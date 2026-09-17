import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config.settings import MODEL, SFT, TRAINING
from model import MiniLLM
from train import AdamWNoLerp, resolve_device

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class SFTDataset(Dataset):
    """PyTorch Dataset for Supervised Fine-Tuning with loss-masked labels."""

    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"SFT dataset '{npz_path}' not found.\n"
                "Run 'python create_sft_dataset.py' to generate it."
            )
        data = np.load(npz_path)
        self.input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
        self.labels = torch.tensor(data["labels"], dtype=torch.long)

        if len(self.input_ids) != len(self.labels):
            raise ValueError("input_ids and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.labels[idx]


def load_base_model(checkpoint_path: str, device: torch.device) -> MiniLLM:
    """Initialize MiniLLM architecture and load pretrained weights."""
    model = MiniLLM(
        vocab_size=MODEL["vocab_size"],
        embedding_dim=MODEL["embedding_dim"],
        num_heads=MODEL["num_heads"],
        num_layers=MODEL["num_layers"],
        max_seq_len=MODEL["max_seq_len"],
        ffn_dim=MODEL["ffn_dim"],
        dropout=MODEL.get("dropout", 0.1),
    )

    if os.path.exists(checkpoint_path):
        print(f"Loading base weights from '{checkpoint_path}'...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        elif isinstance(ckpt, torch.nn.Module):
            state_dict = ckpt.state_dict()
        else:
            raise ValueError(f"Unrecognized checkpoint format in {checkpoint_path}")

        model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully.")
    else:
        print(f"Warning: Base weights '{checkpoint_path}' not found. Training from scratch.")

    return model.to(device)


def run_finetuning(
    dataset_path: str = SFT["output_dataset_path"],
    base_model_path: str = TRAINING["final_model_path"],
    output_model_path: str = SFT["finetuned_model_path"],
    checkpoint_path: str = SFT.get("checkpoint_path", "sft_checkpoint.pth"),
    epochs: int = SFT["epochs"],
    batch_size: int = SFT["batch_size"],
    lr: float = SFT["learning_rate"],
    val_split: float = SFT["val_split"],
):
    device = resolve_device()
    print(f"Using compute device: {device}")

    # 1. Dataset & DataLoader
    dataset = SFTDataset(dataset_path)
    total_len = len(dataset)
    val_len = max(1, int(total_len * val_split))
    train_len = total_len - val_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    print(f"Dataset split: {train_len:,} train samples, {val_len:,} validation samples.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # 2. Model & Optimizer
    if not os.path.exists(base_model_path):
        alt = TRAINING.get("checkpoint_path", "minillm_checkpoint.pth")
        if os.path.exists(alt):
            base_model_path = alt

    model = load_base_model(base_model_path, device)
    optimizer = AdamWNoLerp(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")

    print(f"\nStarting Supervised Fine-Tuning ({epochs} epochs, lr={lr}, batch_size={batch_size})...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits.view(-1, MODEL["vocab_size"]), y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"train_loss": f"{running_loss / train_steps:.4f}"})

        avg_train_loss = running_loss / max(1, train_steps)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, MODEL["vocab_size"]), y.view(-1))
                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)
        print(
            f"Epoch {epoch}/{epochs} Completed: "
            f"Avg Train Loss = {avg_train_loss:.4f} | "
            f"Avg Val Loss = {avg_val_loss:.4f}"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            checkpoint_path,
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model_path)
            print(f"  --> Saved new best fine-tuned model to '{output_model_path}' (Val Loss: {best_val_loss:.4f})")

    print("\nFine-tuning complete!")
    print(f"Final fine-tuned weights saved to: '{os.path.abspath(output_model_path)}'")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniLLM on conversational instruction data.")
    parser.add_argument("--dataset", default=SFT["output_dataset_path"])
    parser.add_argument("--base-model", default=TRAINING["final_model_path"])
    parser.add_argument("--output", default=SFT["finetuned_model_path"])
    parser.add_argument("--checkpoint", default=SFT.get("checkpoint_path", "sft_checkpoint.pth"))
    parser.add_argument("--epochs", type=int, default=SFT["epochs"])
    parser.add_argument("--batch-size", type=int, default=SFT["batch_size"])
    parser.add_argument("--lr", type=float, default=SFT["learning_rate"])
    parser.add_argument("--val-split", type=float, default=SFT["val_split"])
    args = parser.parse_args()

    run_finetuning(
        dataset_path=args.dataset,
        base_model_path=args.base_model,
        output_model_path=args.output,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
