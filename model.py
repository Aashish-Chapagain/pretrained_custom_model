import numpy as np
import torch
import torch.nn as nn
from torch.nn import Embedding, Linear, Module, TransformerDecoder, TransformerDecoderLayer

DEFAULT_VOCAB_SIZE = 5000
DEFAULT_EMBEDDING_DIM = 256
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_LAYERS = 4
DEFAULT_MAX_SEQ_LEN = 256





class MiniLLM(Module):
    def __init__(
        self,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_layers: int = DEFAULT_NUM_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super(MiniLLM, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(embedding_dim, max_seq_len),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            self._generate_causal_mask(max_seq_len),
            persistent=False,
        )
        decoder_layer = TransformerDecoderLayer(d_model = embedding_dim, nhead = num_heads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers = num_layers)
        self.output_layer = Linear(embedding_dim, vocab_size)

    

    def _generate_positional_encoding(self, embedding_dim : int, max_len : int) -> torch.Tensor:
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0)/ embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _generate_causal_mask(self, max_len: int) -> torch.Tensor:
        mask = torch.full((max_len, max_len), float("-inf"))
        return torch.triu(mask, diagonal=1)
    



    def forward(self, x : torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[: seq_len, :].to(x.device)
        x.transpose_(0,1)
        attn_mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        output = self.transformer_decoder(x, x, tgt_mask=attn_mask, memory_mask=attn_mask)
        output = self.output_layer(output)
        return output
    

if __name__ == "__main__":
    model = MiniLLM()
    input_seq = torch.randint(0, DEFAULT_VOCAB_SIZE, (1, 10))
    output = model(input_seq)
    # print(output.shape)