import torch
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    Module,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn import init

from config.settings import MODEL


class MiniLLM(Module):
    def __init__(
        self,
        vocab_size: int = MODEL["vocab_size"],
        embedding_dim: int = MODEL["embedding_dim"],
        num_heads: int = MODEL["num_heads"],
        num_layers: int = MODEL["num_layers"],
        max_seq_len: int = MODEL["max_seq_len"],
        ffn_dim: int = MODEL["ffn_dim"],
        dropout: float = MODEL["dropout"],
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
        self.output_layer.weight = self.embedding.weight
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


if __name__ == "__main__":
    model = MiniLLM()
    input_seq = torch.randint(0, MODEL["vocab_size"], (1, 10))
    output = model(input_seq)
    print(output.shape)
