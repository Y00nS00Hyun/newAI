"""TextCNN text classifier."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN 모델 - 다양한 n-gram 패턴을 포착하여 가짜뉴스 판별"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 128,
        filter_sizes: list = None,
        num_filters: int = 100,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout * 0.25)

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, padding=0)
            for fs in filter_sizes
        ])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_filters) for _ in filter_sizes])

        self.dropout = nn.Dropout(dropout)

        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.BatchNorm1d(total_filters // 2),
            nn.Dropout(dropout * 0.5),
            nn.Linear(total_filters // 2, total_filters // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(total_filters // 4, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Forward pass"""
        embedded = self.embedding_dropout(
            self.embedding(input_ids)).transpose(1, 2)

        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = F.relu(bn(conv(embedded)))
            pooled = F.max_pool1d(
                conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        concatenated = self.dropout(torch.cat(conv_outputs, dim=1))
        return self.classifier(concatenated)
