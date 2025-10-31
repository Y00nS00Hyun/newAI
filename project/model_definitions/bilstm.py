"""Bidirectional LSTM text classifier."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        # Concatenate final states from both directions
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

