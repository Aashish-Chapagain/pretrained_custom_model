import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import DATASET


class LLMDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        sequence_length: int = DATASET["sequence_length"],
        stride: int = DATASET["stride"],
    ) -> None:
        self.tokens = torch.tensor(np.load(file_path), dtype = torch.long)
        self.sequence_length = sequence_length
        self.stride = stride
    

    def __len__(self) -> int:
        return (len(self.tokens) - self.sequence_length-1) // self.stride + 1
    

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length


        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]

        return x, y 
    
    

    

    



    



# dataset = LLMDataset(DATASET["path"], sequence_length=DATASET["sequence_length"])
# print(f"Dataset size: {len(dataset)} sequences")
 
# x, y = dataset[0]

# print(x.shape)
# print(y.shape)