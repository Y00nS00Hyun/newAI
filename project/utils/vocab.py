"""Simple whitespace vocabulary tooling."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TITLE_START = "<title>"
TITLE_END = "</title>"
TEXT_START = "<text>"
TEXT_END = "</text>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, TITLE_START, TITLE_END, TEXT_START, TEXT_END]


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return text.strip().split()


@dataclass
class Vocab:
    max_size: int = 20000
    min_freq: int = 1

    def __post_init__(self) -> None:
        self.token_to_idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.idx_to_token = list(SPECIAL_TOKENS)
        self._frozen = False

    def build(self, corpus: Iterable[str], tokenizer: WhitespaceTokenizer) -> None:
        counter: Counter = Counter()
        for text in corpus:
            counter.update(tokenizer.tokenize(text))
        most_common = counter.most_common(self.max_size - len(self.idx_to_token))
        for token, freq in most_common:
            if freq < self.min_freq:
                continue
            if token in self.token_to_idx:
                continue
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)
        self._frozen = True

    def encode(self, text: str, tokenizer: WhitespaceTokenizer) -> List[int]:
        tokens = tokenizer.tokenize(text)
        return [self.token_to_idx.get(tok, self.token_to_idx[UNK_TOKEN]) for tok in tokens]

    def __len__(self) -> int:
        return len(self.idx_to_token)


def pad_sequences(sequences: Sequence[List[int]], max_len: int, pad_idx: int = 0) -> torch.Tensor:
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if length:
            padded[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
    return padded


def collate_batch(batch, max_len: int, pad_idx: int = 0):
    input_ids = [item["input_ids"] for item in batch]
    lengths = torch.tensor([min(len(ids), max_len) for ids in input_ids], dtype=torch.long)
    padded = pad_sequences(input_ids, max_len=max_len)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return padded, lengths, labels

