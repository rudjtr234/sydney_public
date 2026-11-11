# src/models/layers.py
# Thyroid_Mutation_model_v2/src/models/layers.py 에서 MIL 모델들이 쓰는 부분만 옮김

from typing import List

import torch
import torch.nn as nn


def create_mlp(in_dim: int,
               hid_dims: List[int],
               dropout: float,
               out_dim: int,
               end_with_fc: bool = False) -> nn.Module:
    """
    Create a Multi-Layer Perceptron (MLP) with specified dimensions.

    Args:
        in_dim (int): Input dimension
        hid_dims (List[int]): List of hidden layer dimensions
        dropout (float): Dropout probability
        out_dim (int): Output dimension
        end_with_fc (bool): If True, end with only linear layer without activation
    """
    layers = []
    dims = [in_dim] + hid_dims + [out_dim]

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or not end_with_fc:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class GlobalAttention(nn.Module):
    """Standard (tanh) attention pooling for MIL."""

    def __init__(self, L: int, D: int, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Linear(D, num_classes)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, N, L] -> attention scores [B, N, num_classes]"""
        return self.attention(self.dropout(h))


class GlobalGatedAttention(nn.Module):
    """Gated attention (tanh * sigmoid) pooling for MIL."""

    def __init__(self, L: int, D: int, dropout: float = 0.0, num_classes: int = 1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_w = nn.Linear(D, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, N, L] -> attention scores [B, N, num_classes]"""
        h = self.dropout(h)
        return self.attention_w(self.attention_V(h) * self.attention_U(h))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based MIL models."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [seq_len, B, d_model]"""
        return x + self.pe[:x.size(0), :]
