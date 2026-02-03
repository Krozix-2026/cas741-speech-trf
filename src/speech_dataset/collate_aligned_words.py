# speech_dataset/collate_aligned_words.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

IGNORE = -100

@dataclass
class AlignedWordBatch:
    feats: torch.Tensor      # (B, Tmax, 64)
    feat_lens: torch.Tensor  # (B,)
    targets: torch.Tensor    # (B, Tmax)
    utt_ids: List[str]

def collate_aligned_words(items: List[Dict[str, Any]]) -> AlignedWordBatch:
    items = sorted(items, key=lambda x: x["feat_len"], reverse=True)
    B = len(items)
    lens = torch.tensor([it["feat_len"] for it in items], dtype=torch.long)
    Tmax = int(lens.max().item())
    F = int(items[0]["feat"].shape[1])

    feats = torch.zeros(B, Tmax, F, dtype=torch.float32)
    targets = torch.full((B, Tmax), IGNORE, dtype=torch.long)

    for i, it in enumerate(items):
        t = it["feat_len"]
        feats[i, :t] = it["feat"]
        targets[i, :t] = it["target"]

    utt_ids = [it["utt_id"] for it in items]
    return AlignedWordBatch(feats=feats, feat_lens=lens, targets=targets, utt_ids=utt_ids)