"""TextCNN text classifier."""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN 모델 (Kim 2014 스타일)

    여러 필터 크기를 사용하여 다양한 n-gram 패턴을 포착합니다.
    가짜뉴스 판별에 유용한 지역적 패턴을 잘 학습할 수 있습니다.
    """

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

        # 기본 필터 크기 설정 (3, 4, 5-gram)
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(
            dropout * 0.25)  # Embedding dropout (약간 낮춤)

        # 각 필터 크기에 대해 Conv1d 레이어 생성
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding=0
            )
            for filter_size in filter_sizes
        ])

        # BatchNorm 추가로 학습 안정화 및 성능 향상
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        # 모든 필터를 concatenate한 후 분류
        # filter_sizes 개수만큼 num_filters를 concatenate
        total_filters = num_filters * len(filter_sizes)

        # 2층 분류기로 변경 (더 복잡한 패턴 학습)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(total_filters // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) 형태의 토큰 인덱스
            lengths: 각 시퀀스의 실제 길이 (CNN에서는 사용하지 않지만 호환성을 위해 유지)

        Returns:
            logits: (batch_size, num_classes) 형태의 분류 로그its
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)  # Embedding dropout 적용

        # Conv1d는 (batch, channels, length) 형태를 요구하므로 transpose
        # (batch_size, seq_len, embedding_dim) -> (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # 각 필터 크기에 대해 convolution 적용 후 max pooling
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # Convolution: (batch_size, embedding_dim, seq_len) -> (batch_size, num_filters, new_seq_len)
            conv_out = conv(embedded)
            conv_out = bn(conv_out)  # BatchNorm 적용
            conv_out = F.relu(conv_out)

            # Max pooling over time: (batch_size, num_filters, new_seq_len) -> (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)

        # 모든 필터 출력을 concatenate
        # (batch_size, num_filters * len(filter_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)

        # Dropout 적용
        concatenated = self.dropout(concatenated)

        # 분류
        logits = self.classifier(concatenated)
        return logits
