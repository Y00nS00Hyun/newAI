"""Dataset loading utilities for fake news datasets."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from utils import vocab as vocab_lib
from utils.constants import LABEL_TO_IDX


class FakeNewsDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_fields: List[str],
        tokenizer: vocab_lib.WhitespaceTokenizer,
        vocab: vocab_lib.Vocab,
        max_len: int,
    ) -> None:
        if dataframe.empty:
            raise ValueError("Dataset received an empty DataFrame")
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.text_fields = text_fields
        self.texts = [
            compose_record_text(row, self.text_fields)
            for _, row in dataframe.iterrows()
        ]
        # 문자열 label을 숫자로 변환 (학습에는 정수 레이블 필요)
        self.labels = dataframe["label"].map(LABEL_TO_IDX).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        token_ids = self.vocab.encode(text, self.tokenizer)
        return {"input_ids": token_ids, "label": self.labels[idx]}


def compose_record_text(row: pd.Series, text_fields: List[str]) -> str:
    def wrap(tag: str, content: str) -> str:
        stripped = content.strip()
        if stripped:
            return f"<{tag}> {stripped} </{tag}>"
        return f"<{tag}> </{tag}>"

    title_value = ""
    text_value = ""
    if "title" in text_fields:
        value = row.get("title")
        if pd.notna(value):
            title_value = str(value)
    if "text" in text_fields:
        value = row.get("text")
        if pd.notna(value):
            text_value = str(value)

    return f"{wrap('title', title_value)} {wrap('text', text_value)}".strip()


def read_csv_file(path: Path) -> pd.DataFrame:
    """CSV 파일을 읽어서 DataFrame으로 반환"""
    # 다양한 구분자 시도
    try:
        df = pd.read_csv(path, encoding='utf-8', sep=',')
    except:
        try:
            df = pd.read_csv(path, encoding='utf-8', sep=';')
        except:
            df = pd.read_csv(path, encoding='utf-8', sep=',', engine='python', quoting=3)
    
    # 컬럼명 정리
    df.columns = [col.strip() for col in df.columns]
    
    # Unnamed 컬럼 제거
    drop_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    return df


def build_dataloaders(
    train_path: Path,
    val_path: Path,
    batch_size: int,
    max_len: int,
    num_workers: int = 0,
    max_vocab_size: int = 20000,
    text_fields: List[str] = None,
):
    """학습 및 검증 데이터 로더 생성"""
    if text_fields is None:
        text_fields = ["title", "text"]
    
    # CSV 파일 읽기
    train_df = read_csv_file(train_path)
    val_df = read_csv_file(val_path)
    
    # Tokenizer와 Vocab 생성
    tokenizer = vocab_lib.WhitespaceTokenizer()
    vocab = vocab_lib.Vocab(max_size=max_vocab_size)
    
    # Vocab 빌드 (train 데이터로)
    train_texts = [
        compose_record_text(row, text_fields) 
        for _, row in train_df.iterrows()
    ]
    vocab.build(train_texts, tokenizer)
    
    # Dataset 생성
    train_dataset = FakeNewsDataset(train_df, text_fields, tokenizer, vocab, max_len)
    val_dataset = FakeNewsDataset(val_df, text_fields, tokenizer, vocab, max_len)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: vocab_lib.collate_batch(batch, max_len=max_len),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: vocab_lib.collate_batch(batch, max_len=max_len),
    )
    
    return train_loader, val_loader, vocab, tokenizer

