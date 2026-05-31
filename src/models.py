"""
Neural models for elution sequence prediction.

Architecture: Multi-field embedding → sequence encoder → classification head

Models:
  1. LSTM: Multi-field embeddings → LSTM → FC head
  2. Transformer: Multi-field embeddings → causal self-attention → FC head

Input: 5 parallel integer sequences (mz_bin, md_bin, rt_gap, polarity, intensity)
Output: probability distribution over m/z bins (next-token prediction)
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    NUM_HEADS,
    FF_DIM,
    CONTEXT_LENGTH,
)


class MultiFieldEmbedding(nn.Module):
    """Embed each token field separately, then combine.

    Fields: mz_bin, md_bin, rt_gap_idx, polarity_idx, intensity_idx
    Each gets its own embedding table. Outputs are summed (not concatenated)
    to keep dimensionality manageable.
    """

    def __init__(self, embedding_dim, max_mz_bin=120, max_md_bin=20,
                 max_rt_gap=7, max_polarity=3, max_intensity=5):
        super().__init__()
        self.mz_embed = nn.Embedding(max_mz_bin, embedding_dim)
        self.md_embed = nn.Embedding(max_md_bin, embedding_dim)
        self.gap_embed = nn.Embedding(max_rt_gap, embedding_dim)
        self.pol_embed = nn.Embedding(max_polarity, embedding_dim)
        self.int_embed = nn.Embedding(max_intensity, embedding_dim)

    def forward(self, mz, md, gap, pol, intensity):
        return (
            self.mz_embed(mz)
            + self.md_embed(md)
            + self.gap_embed(gap)
            + self.pol_embed(pol)
            + self.int_embed(intensity)
        )


class LSTMModel(nn.Module):
    """LSTM-based sequence model for next-token prediction."""

    def __init__(self, num_mz_classes, embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, **embed_kwargs):
        super().__init__()
        self.embedding = MultiFieldEmbedding(embedding_dim, **embed_kwargs)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_mz_classes)

    def forward(self, mz, md, gap, pol, intensity):
        x = self.embedding(mz, md, gap, pol, intensity)  # (B, T, E)
        out, _ = self.lstm(x)  # (B, T, H)
        last = out[:, -1, :]  # Take last time step
        last = self.dropout(last)
        return self.head(last)  # (B, num_classes)


class TransformerModel(nn.Module):
    """Transformer decoder for next-token prediction (causal attention)."""

    def __init__(self, num_mz_classes, embedding_dim=EMBEDDING_DIM,
                 num_heads=NUM_HEADS, ff_dim=FF_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT,
                 context_length=CONTEXT_LENGTH, **embed_kwargs):
        super().__init__()
        self.embedding = MultiFieldEmbedding(embedding_dim, **embed_kwargs)
        self.pos_embed = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.head = nn.Linear(embedding_dim, num_mz_classes)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, mz, md, gap, pol, intensity):
        B, T = mz.shape
        x = self.embedding(mz, md, gap, pol, intensity)  # (B, T, E)

        # Add positional embedding
        pos = torch.arange(T, device=mz.device).unsqueeze(0)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)

        # Causal self-attention
        mask = self.causal_mask[:T, :T]
        x = self.transformer(x, mask=mask)  # (B, T, E)

        last = x[:, -1, :]  # Take last position
        return self.head(last)  # (B, num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test with dummy data
    B, T = 4, 64
    num_classes = 120

    mz = torch.randint(0, 120, (B, T))
    md = torch.randint(0, 20, (B, T))
    gap = torch.randint(0, 7, (B, T))
    pol = torch.randint(0, 3, (B, T))
    inten = torch.randint(0, 5, (B, T))

    print("=== LSTM Model ===")
    lstm = LSTMModel(num_classes)
    out = lstm(mz, md, gap, pol, inten)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(lstm):,}")

    print("\n=== Transformer Model ===")
    transformer = TransformerModel(num_classes)
    out = transformer(mz, md, gap, pol, inten)
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {count_parameters(transformer):,}")
