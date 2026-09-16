import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.settings import DATASET, MODEL, TRAINING
from datasetclass import LLMDataset
from model import MiniLLM

try:
    import torch_directml
    _HAS_DIRECTML = True
except Exception:
    _HAS_DIRECTML = False


class AdamWNoLerp(optim.Optimizer):
    """AdamW that avoids aten::lerp, which DirectML does not support."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure: Optional[object] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss


def resolve_device() -> torch.device:
    if _HAS_DIRECTML:
        torch.backends.mha.set_fastpath_enabled(False)
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


CHECKPOINT_PATH = TRAINING["checkpoint_path"]
FINAL_MODEL_PATH = TRAINING["final_model_path"]
EPOCH_SAVE_INTERVAL = TRAINING["epoch_save_interval"]
VOCAB_SIZE = MODEL["vocab_size"]
DATASET_PATH = DATASET["path"]
SEQUENCE_LENGTH = DATASET["sequence_length"]
STRIDE = DATASET["stride"]
LEARNING_RATE = TRAINING["learning_rate"]
EPOCHS = TRAINING["epochs"]
BATCH_SIZE = TRAINING["batch_size"]
RESUME = TRAINING["resume"]


def _try_resume(model: MiniLLM, optimizer: optim.Optimizer) -> int:
    if not RESUME or not os.path.exists(CHECKPOINT_PATH):
        return 0

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, KeyError) as exc:
        print(f"Checkpoint is incompatible, starting from scratch: {exc}")
        return 0

    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"Optimizer state not restored: {exc}")

    start_epoch = int(checkpoint.get("epoch", 0))
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch


def train(
    model: MiniLLM,
    dataset_path: str = DATASET_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> None:
    device = resolve_device()
    print(f"Using device: {device}")

    dataset = LLMDataset(
        dataset_path,
        sequence_length=sequence_length,
        stride=stride,
    )
    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )

    model.to(device)
    if _HAS_DIRECTML:
        optimizer = AdamWNoLerp(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    start_epoch = _try_resume(model, optimizer)
    model.train()

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(dataloader))

        if (epoch + 1) % EPOCH_SAVE_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "vocab_size": VOCAB_SIZE,
                },
                CHECKPOINT_PATH,
            )
            print(f"checkpoint saved at epoch({epoch + 1})")

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)


if __name__ == "__main__":
    train(MiniLLM())
