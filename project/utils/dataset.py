"""Dataset loading utilities for fake news datasets."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from utils import vocab as vocab_lib
from utils.constants import LABEL_TO_IDX

SPECIAL_WRAP_TOKENS = {
    "title": ("<title>", "</title>"),
    "text": ("<text>", "</text>"),
}

# -----------------------------
# Augmentation
# -----------------------------


@dataclass
class TextAugmentConfig:
    p_delete: float = 0.15  # 문장 단위로 삭제 연산을 시도할 확률
    p_swap: float = 0.15    # 문장 단위로 스왑 연산을 시도할 확률
    keep_ratio_if_delete: float = 0.85  # 삭제 시 개별 단어 유지 확률
    respect_wrapped_tokens: bool = True  # <title> ... </title> 등 보호
    seed: Optional[int] = None  # 재현성


class TextAugmenter:
    """텍스트 증강기. 재현성을 위해 선택적으로 seed 고정 지원."""

    def __init__(self, cfg: TextAugmentConfig):
        self.cfg = cfg
        self._rng = random.Random(cfg.seed) if cfg.seed is not None else random

    def _is_wrapped_token(self, w: str) -> bool:
        if not self.cfg.respect_wrapped_tokens:
            return False
        return w.startswith("<") and w.endswith(">")

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        # 1) 삭제
        if self._rng.random() < self.cfg.p_delete:
            kept: List[str] = []
            for w in words:
                if self._is_wrapped_token(w):
                    kept.append(w)
                else:
                    if self._rng.random() < self.cfg.keep_ratio_if_delete:
                        kept.append(w)
            # 비어버리면 원문 유지
            words = kept or words

        # 2) 스왑
        if self._rng.random() < self.cfg.p_swap and len(words) > 2:
            normal_idx = [i for i, w in enumerate(
                words) if not self._is_wrapped_token(w)]
            if len(normal_idx) >= 2:
                i, j = self._rng.sample(normal_idx, 2)
                words[i], words[j] = words[j], words[i]

        return " ".join(words)

# -----------------------------
# Text utilities
# -----------------------------


def _wrap(tag: str, content: str) -> str:
    """빈 문자열도 안전하게 래핑."""
    left, right = SPECIAL_WRAP_TOKENS.get(tag, (f"<{tag}>", f"</{tag}>"))
    content = (content or "").strip()
    return f"{left} {content} {right}".strip()


def compose_record_text(row: pd.Series, text_fields: Sequence[str]) -> str:
    """여러 필드를 래핑하여 하나의 문자열로 합침."""
    parts: List[str] = []
    for field in text_fields:
        val = row.get(field)
        val = "" if (pd.isna(val) or val is None) else str(val)
        parts.append(_wrap(field, val))
    return " ".join(parts).strip()

# -----------------------------
# CSV loading
# -----------------------------


def read_csv_file(path: Path) -> pd.DataFrame:
    """
    CSV 로더: 인코딩/구분자 자동 시도 + Unnamed 컬럼 제거 + 컬럼명 트리밍.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    tried: List[Tuple[str, str]] = []
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        for sep in (",", ";", "\t"):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                tried.append((enc, sep))
                break
            except Exception:
                continue
        else:
            continue
        break
    else:
        # 마지막 시도로 python engine 사용
        df = pd.read_csv(path, encoding="utf-8", sep=",",
                         engine="python", quoting=csv.QUOTE_MINIMAL)

    # 컬럼 정리
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if df.empty:
        raise ValueError(f"CSV is empty after cleaning: {path}")

    return df

# -----------------------------
# Dataset
# -----------------------------


class FakeNewsDataset(Dataset):
    """
    텍스트 필드를 래핑하여 하나의 시퀀스로 만들고, vocab/tokenizer로 인코딩.
    학습 데이터인 경우 Augmenter를 적용.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_fields: Sequence[str],
        tokenizer: vocab_lib.WhitespaceTokenizer,
        vocab: vocab_lib.Vocab,
        max_len: int,
        is_train: bool = False,
        augmenter: Optional[TextAugmenter] = None,
    ) -> None:
        if dataframe is None or dataframe.empty:
            raise ValueError("Dataset received an empty DataFrame")
        if "label" not in dataframe.columns:
            raise KeyError("Dataset requires a 'label' column")

        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = int(max_len)
        self.text_fields = list(text_fields)
        self.is_train = bool(is_train)
        self.augmenter = augmenter

        # 미리 합성
        self.texts: List[str] = [
            compose_record_text(row, self.text_fields) for _, row in dataframe.iterrows()
        ]

        # 라벨 인덱싱 + 검증
        try:
            self.labels: List[int] = dataframe["label"].map(
                LABEL_TO_IDX).astype(int).tolist()
        except Exception as e:
            unknown = set(dataframe["label"]) - set(LABEL_TO_IDX.keys())
            raise ValueError(f"Unknown labels detected: {unknown}") from e

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]

        if self.is_train and self.augmenter is not None:
            text = self.augmenter(text)

        token_ids = self.vocab.encode(text, self.tokenizer)
        return {"input_ids": token_ids, "label": self.labels[idx]}

# -----------------------------
# Dataloader builder
# -----------------------------


@dataclass
class DataBuildConfig:
    train_path: Path
    val_path: Path
    batch_size: int = 64
    max_len: int = 512
    num_workers: int = 0
    max_vocab_size: int = 20_000
    text_fields: Tuple[str, ...] = ("title", "text")
    augment: bool = True
    augment_seed: Optional[int] = 42
    p_delete: float = 0.15
    p_swap: float = 0.15
    keep_ratio_if_delete: float = 0.85


def _collate(vocab: vocab_lib.Vocab, batch: List[dict], max_len: int):
    return vocab_lib.collate_batch(batch, max_len=max_len)


def build_dataloaders_from_config(cfg: DataBuildConfig):
    """
    학습/검증 DataLoader와 vocab/tokenizer를 생성.

    Returns:
        train_loader, val_loader, vocab, tokenizer
    """
    # 1) CSV 읽기
    train_df = read_csv_file(Path(cfg.train_path))
    val_df = read_csv_file(Path(cfg.val_path))

    # 2) Tokenizer/Vocab
    tokenizer = vocab_lib.WhitespaceTokenizer()
    vocab = vocab_lib.Vocab(max_size=cfg.max_vocab_size)

    # 3) Vocab build (train 텍스트로만)
    train_texts = [compose_record_text(
        row, cfg.text_fields) for _, row in train_df.iterrows()]
    vocab.build(train_texts, tokenizer)

    # 4) Augmenter
    augmenter = None
    if cfg.augment:
        aug_cfg = TextAugmentConfig(
            p_delete=cfg.p_delete,
            p_swap=cfg.p_swap,
            keep_ratio_if_delete=cfg.keep_ratio_if_delete,
            seed=cfg.augment_seed,
        )
        augmenter = TextAugmenter(aug_cfg)

    # 5) Datasets
    train_ds = FakeNewsDataset(
        dataframe=train_df,
        text_fields=cfg.text_fields,
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=cfg.max_len,
        is_train=True,
        augmenter=augmenter,
    )

    val_ds = FakeNewsDataset(
        dataframe=val_df,
        text_fields=cfg.text_fields,
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=cfg.max_len,
        is_train=False,
        augmenter=None,
    )

    # 6) DataLoaders
    collate_fn = partial(_collate, vocab, max_len=cfg.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, vocab, tokenizer

# -----------------------------
# 기존 호환성을 위한 wrapper 함수
# -----------------------------


def build_dataloaders(
    train_path: Path,
    val_path: Path,
    batch_size: int,
    max_len: int,
    num_workers: int = 0,
    max_vocab_size: int = 20000,
    text_fields: List[str] = None,
):
    """
    기존 호환성을 유지하는 wrapper 함수.

    Args:
        train_path: 학습 데이터 경로
        val_path: 검증 데이터 경로
        batch_size: 배치 크기
        max_len: 최대 시퀀스 길이
        num_workers: 데이터 로더 워커 수
        max_vocab_size: 최대 어휘 크기
        text_fields: 텍스트 필드 리스트

    Returns:
        train_loader, val_loader, vocab, tokenizer
    """
    if text_fields is None:
        text_fields = ["title", "text"]

    cfg = DataBuildConfig(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
        max_vocab_size=max_vocab_size,
        text_fields=tuple(text_fields),
        augment=True,  # 기본적으로 증강 활성화
        p_delete=0.12,  # 증강 강도 약간 낮춤 (원본 데이터에 더 가깝게)
        p_swap=0.12,
        keep_ratio_if_delete=0.88,  # 단어 유지 비율 증가
    )

    return build_dataloaders_from_config(cfg)
