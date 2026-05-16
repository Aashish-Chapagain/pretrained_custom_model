import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_STRIDE = 64


class LLMDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        stride: int = DEFAULT_STRIDE,
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
    
    

    

    



    



# dataset = LLMDataset("tokenized_corpus.npy", sequence_length=256 )
# print(f"Dataset size: {len(dataset)} sequences")
 
# x, y = dataset[0]

# print(x.shape)
# print(y.shape)