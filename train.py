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


def resolve_device() -> torch.device:
    if _HAS_DIRECTML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


device = resolve_device()
print(f"Using device: {device}")

CHECKPOINT_PATH = TRAINING["checkpoint_path"]
EPOCH_SAVE_INTERVAL = TRAINING["epoch_save_interval"]
VOCAB_SIZE = MODEL["vocab_size"]
DATASET_PATH = DATASET["path"]
SEQUENCE_LENGTH = DATASET["sequence_length"]
STRIDE = DATASET["stride"]
LEARNING_RATE = TRAINING["learning_rate"]
EPOCHS = TRAINING["epochs"]
BATCH_SIZE = TRAINING["batch_size"]

def train(
    model: MiniLLM,
    dataset_path: str = DATASET_PATH,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = STRIDE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> None:
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            output = output.transpose(0, 1).contiguous()
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(dataloader))


        
        if (epoch + 1) % EPOCH_SAVE_INTERVAL == 0:
            torch.save({
                "epoch" :epoch + 1, 
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : avg_loss,
                "Vocab_size" : VOCAB_SIZE
            }, CHECKPOINT_PATH)

            print(f"checkpoint saved at epoch({epoch + 1})")

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    model = MiniLLM(
        vocab_size=MODEL["vocab_size"],
        embedding_dim=MODEL["embedding_dim"],
        num_heads=MODEL["num_heads"],
        num_layers=MODEL["num_layers"],
        max_seq_len=MODEL["max_seq_len"],
    )
    train(model, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)