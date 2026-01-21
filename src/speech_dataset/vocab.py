# speech_dataset/vocab.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


def normalize_text(s: str) -> str:
    # LibriSpeech transcript is uppercase; keep [a-z ' ] + space
    s = s.lower()
    s = re.sub(r"[^a-z' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass(frozen=True)
class CharVocab:
    """
    CTC vocab: blank=0, then tokens.
    tokens include space and apostrophe for LibriSpeech.
    """
    tokens: List[str]

    @property
    def blank_idx(self) -> int:
        return 0

    @property
    def stoi(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.tokens)}

    @property
    def itos(self) -> Dict[int, str]:
        return {i: c for i, c in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        text = normalize_text(text)
        table = self.stoi
        # skip unknowns silently
        return [table[c] for c in text if c in table]

    def decode_ids(self, ids: List[int]) -> str:
        table = self.itos
        s = "".join(table[i] for i in ids if i in table)
        s = normalize_text(s)
        return s


def build_librispeech_char_vocab() -> CharVocab:
    # blank + space + apostrophe + a..z
    tokens = ["<blank>", " ", "'"] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
    return CharVocab(tokens=tokens)
