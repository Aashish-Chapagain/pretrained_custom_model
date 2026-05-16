import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasetclass import LLMDataset
from model import MiniLLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "minillm_checkpoint.pth"
EPOCH_SAVE_INTERVAL = 20
VOCAB_SIZE = 5000
DATASET_PATH = "tokenized_corpus.npy"
SEQUENCE_LENGTH = 256
STRIDE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 16

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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    model = MiniLLM(vocab_size=VOCAB_SIZE, max_seq_len=SEQUENCE_LENGTH)
    train(model, sequence_length=SEQUENCE_LENGTH, stride=STRIDE)