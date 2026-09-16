import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import DATASET, MODEL


class LLMDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        sequence_length: int = DATASET["sequence_length"],
        stride: int = DATASET["stride"],
        vocab_size: int = MODEL["vocab_size"],
    ) -> None:
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.tokens = torch.tensor(np.load(file_path), dtype=torch.long)
        if self.tokens.ndim != 1:
            raise ValueError(f"Token array must be 1D, got shape {tuple(self.tokens.shape)}")
        if self.tokens.numel() < sequence_length:
            raise ValueError(
                f"Token dataset is too short for sequence_length={sequence_length}: "
                f"got {self.tokens.numel()} tokens"
            )

        min_token = int(self.tokens.min().item())
        max_token = int(self.tokens.max().item())
        if min_token < 0 or max_token >= vocab_size:
            raise ValueError(
                f"Token values are out of range for vocab_size={vocab_size}: "
                f"min={min_token}, max={max_token}"
            )

        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        if self.tokens.numel() < self.sequence_length:
            return 0
        return (self.tokens.numel() - self.sequence_length) // self.stride + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]
        return x, y
