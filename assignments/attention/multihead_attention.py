"""
Masked Multi-Head Attention Assignment (Student Version)
========================================================

Implement masked multi-head attention in pure PyTorch.

Reference: https://arxiv.org/pdf/1706.03762

Goal:
- project query/key/value
- split into heads
- compute scaled dot-product attention
- apply an attention mask (causal or padding)
- merge heads and project output

Complete all TODOs.
"""

import math

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    """
    Expected shapes:
    - query: (batch, query_len, embed_dim)
    - key:   (batch, key_len, embed_dim)
    - value: (batch, key_len, embed_dim)
    - attn_mask: usually provided for causal/padding constraints
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        # TODO: validate that embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # TODO: create linear projections for q, k, v and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # TODO: create dropout layer for attention weights
        self.attn_dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Convert shape:
        (batch, seq_len, embed_dim) -> (batch, heads, seq_len, head_dim)
        """
        # TODO: reshape and transpose
        batch, seq_len, embed_dim = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Convert shape:
        (batch, heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)
        """
        # TODO: transpose back, then reshape
        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        return x.view(batch, seq_len, self.embed_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Tensor | None = None,
    ) -> Tensor:
        # TODO: project query/key/value and split heads
        q = self._split_heads(self.q_proj(query))  # (batch, heads, query_len, head_dim)
        k = self._split_heads(self.k_proj(key))  # (batch, heads, key_len,   head_dim)
        v = self._split_heads(self.v_proj(value))  # (batch, heads, key_len,   head_dim)

        # TODO: compute scaled dot-product attention scores
        # expected shape: (batch, heads, query_len, key_len)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, query_len, key_len)

        # TODO:
        # Expand attn_mask to match scores shape: (batch, heads, query_len, key_len).
        # This makes the broadcasting that happens in masked_fill below explicit.
        if attn_mask is not None:
            # attn_mask is (query_len, key_len) or (batch, query_len, key_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, query_len, key_len)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (batch, 1, query_len, key_len)
            attn_mask = attn_mask.expand_as(scores)  # (batch, heads, query_len, key_len)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        # Softmax over key dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # TODO: apply dropout to attention weights.
        attn_weights = self.attn_dropout(attn_weights)

        # TODO: multiply attention weights by V to get context
        context = torch.matmul(attn_weights, v)  # (batch, heads, query_len, head_dim)

        # TODO: merge heads and apply output projection
        output = self.out_proj(self._merge_heads(context))

        return output


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4

    # TODO: instantiate multi-head attention.
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # TODO: create causal mask for autoregressive self-attention (torch.bool).
    # True means "masked out" (cannot attend), so upper triangle (future positions) = True
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    # TODO: run forward pass and print output shape
    # Expected shape: (batch_size, seq_len, embed_dim)
    output = mha(x, x, x, attn_mask=causal_mask)
    print(f"Output shape: {output.shape}")  # Expected: (2, 5, 16)

    print("Student assignment executed.")
    print("Complete all TODO sections to produce working multi-head attention.")


if __name__ == "__main__":
    main()
