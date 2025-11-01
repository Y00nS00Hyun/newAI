# -*- coding: utf-8 -*-
import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1d(nn.Module):
    """Squeeze-and-Excitation for 1D features (C x T)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = x.mean(dim=-1, keepdim=True)          # global avg pool over time
        a = F.relu(self.fc1(s))
        a = torch.sigmoid(self.fc2(a))            # (B, C, 1)
        return x * a


class AttentionPool1d(nn.Module):
    """Learnable attention pooling over time."""

    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(channels))  # (C,)
        nn.init.normal_(self.q, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # score_t = <q, x_t>
        q = self.q.view(1, -1, 1)
        score = (x * q).sum(dim=1)                 # (B, T)
        w = torch.softmax(score, dim=-1).unsqueeze(1)  # (B,1,T)
        return (x * w).sum(dim=-1)                 # (B,C)


class SepConvBlock1d(nn.Module):
    """
    Depthwise separable Conv1d + BN + GELU (+ optional dilation) × 2
    with residual (1x1 if needed)
    """

    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) // 2 * dilation

        self.dw1 = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad,
                             dilation=dilation, groups=in_ch, bias=False)
        self.pw1 = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.dw2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad,
                             dilation=dilation, groups=out_ch, bias=False)
        self.pw2 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.se = SE1d(out_ch, reduction=8)
        self.drop = nn.Dropout(dropout)

        self.res_proj = None
        if in_ch != out_ch:
            self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.res_proj is None else self.res_proj(x)
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dw2(x)
        x = self.pw2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x + res


class ImprovedTextCNN(nn.Module):
    """
    업그레이드 TextCNN
    - 멀티커널(예: 3,4,5)
    - 커널별 SepConv 블록(딜레이션 지원)
    - 각 분기에서 (max/avg/attention) 멀티풀링 → concat
    - LayerNorm + Dropout → Linear
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 2,
        embedding_dim: int = 300,
        kernel_sizes: Iterable[int] = (3, 4, 5),
        num_filters_per_kernel: int = 256,
        dilation_cycle: Iterable[int] = (1, 2),   # 예: 1,2 번갈아
        dropout: float = 0.4,
        padding_idx: int = 0,
        embed_dropout: float = 0.2,
        use_pretrained: bool = False,
        pretrained_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        if use_pretrained and pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
        self.embed_drop = nn.Dropout(embed_dropout)

        self.branches = nn.ModuleList()
        self.attn_pools = nn.ModuleList()
        self.kernel_sizes = list(kernel_sizes)

        for k in self.kernel_sizes:
            in_ch = embedding_dim
            out_ch = num_filters_per_kernel
            # 딜레이션을 한 번 혹은 여러 번 적용
            blocks = []
            for i, d in enumerate(dilation_cycle):
                blocks.append(SepConvBlock1d(
                    in_ch if i == 0 else out_ch, out_ch, k=k, dilation=d, dropout=dropout))
            self.branches.append(nn.Sequential(*blocks))
            self.attn_pools.append(AttentionPool1d(out_ch))

        # 멀티풀링: max/avg/attn → 3 × num_filters_per_kernel × #kernels
        pooled_dim = len(self.kernel_sizes) * num_filters_per_kernel * 3
        self.norm = nn.LayerNorm(pooled_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(pooled_dim, num_classes)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # input_ids: (B, T)
        x = self.embedding(input_ids).transpose(1, 2)     # (B, C=emb, T)
        x = self.embed_drop(x)

        feats: List[torch.Tensor] = []
        for branch, attn_pool in zip(self.branches, self.attn_pools):
            h = branch(x)                                  # (B, F, T)
            max_p = F.max_pool1d(h, kernel_size=h.size(-1)
                                 ).squeeze(-1)  # (B,F)
            avg_p = F.adaptive_avg_pool1d(
                h, 1).squeeze(-1)              # (B,F)
            # (B,F)
            att_p = attn_pool(h)
            feats.extend([max_p, avg_p, att_p])

        z = torch.cat(feats, dim=-1)                      # (B, pooled_dim)
        z = self.norm(z)
        z = self.drop(z)
        logits = self.fc(z)                               # (B, num_classes)
        return logits
