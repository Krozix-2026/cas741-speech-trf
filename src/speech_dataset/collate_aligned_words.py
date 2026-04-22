# speech_dataset/collate_aligned_words.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

IGNORE = -100

@dataclass
class AlignedWordBatch:
    feats: torch.Tensor       # (B, Tmax, F)
    feat_lens: torch.Tensor   # (B,)
    targets: torch.Tensor     # (B, Tmax)

    word_ids: torch.Tensor    # (B, Wmax) padded with IGNORE
    word_starts: torch.Tensor # (B, Wmax) padded with -1
    word_ends: torch.Tensor   # (B, Wmax) padded with -1
    word_lens: torch.Tensor   # (B,)

    utt_ids: List[str]

def collate_aligned_words(items: List[Dict[str, Any]]) -> AlignedWordBatch:
    items = sorted(items, key=lambda x: x["feat_len"], reverse=True)
    B = len(items)

    lens = torch.tensor([it["feat_len"] for it in items], dtype=torch.long)
    Tmax = int(lens.max().item())
    F = int(items[0]["feat"].shape[1])

    feats = torch.zeros(B, Tmax, F, dtype=torch.float32)
    targets = torch.full((B, Tmax), IGNORE, dtype=torch.long)

    word_lens = torch.tensor([int(it["word_ids"].numel()) for it in items], dtype=torch.long)
    Wmax = int(word_lens.max().item()) if B > 0 else 0

    word_ids = torch.full((B, Wmax), IGNORE, dtype=torch.long)
    word_starts = torch.full((B, Wmax), -1, dtype=torch.long)
    word_ends = torch.full((B, Wmax), -1, dtype=torch.long)

    for i, it in enumerate(items):
        t = it["feat_len"]
        feats[i, :t] = it["feat"]
        targets[i, :t] = it["target"]

        w = int(it["word_ids"].numel())
        if w > 0:
            word_ids[i, :w] = it["word_ids"]
            word_starts[i, :w] = it["word_starts"]
            word_ends[i, :w] = it["word_ends"]

    utt_ids = [it["utt_id"] for it in items]
    return AlignedWordBatch(
        feats=feats,
        feat_lens=lens,
        targets=targets,
        word_ids=word_ids,
        word_starts=word_starts,
        word_ends=word_ends,
        word_lens=word_lens,
        utt_ids=utt_ids,
    )