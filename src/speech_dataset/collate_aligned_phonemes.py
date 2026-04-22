# speech_dataset/collate_aligned_phonemes.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

IGNORE = -100


@dataclass
class AlignedPhoneBatch:
    feats: torch.Tensor
    feat_lens: torch.Tensor
    targets: torch.Tensor

    phone_ids: torch.Tensor
    phone_starts: torch.Tensor
    phone_ends: torch.Tensor
    phone_lens: torch.Tensor

    utt_ids: List[str]


def collate_aligned_phonemes(items: List[Dict[str, Any]]) -> AlignedPhoneBatch:
    items = sorted(items, key=lambda x: x["feat_len"], reverse=True)
    B = len(items)

    lens = torch.tensor([it["feat_len"] for it in items], dtype=torch.long)
    Tmax = int(lens.max().item())
    F = int(items[0]["feat"].shape[1])

    feats = torch.zeros(B, Tmax, F, dtype=torch.float32)
    targets = torch.full((B, Tmax), IGNORE, dtype=torch.long)

    phone_lens = torch.tensor([int(it["phone_ids"].numel()) for it in items], dtype=torch.long)
    Pmax = int(phone_lens.max().item()) if B > 0 else 0

    phone_ids = torch.full((B, Pmax), IGNORE, dtype=torch.long)
    phone_starts = torch.full((B, Pmax), -1, dtype=torch.long)
    phone_ends = torch.full((B, Pmax), -1, dtype=torch.long)

    for i, it in enumerate(items):
        t = it["feat_len"]
        feats[i, :t] = it["feat"]
        targets[i, :t] = it["target"]

        p = int(it["phone_ids"].numel())
        if p > 0:
            phone_ids[i, :p] = it["phone_ids"]
            phone_starts[i, :p] = it["phone_starts"]
            phone_ends[i, :p] = it["phone_ends"]

    utt_ids = [it["utt_id"] for it in items]
    return AlignedPhoneBatch(
        feats=feats,
        feat_lens=lens,
        targets=targets,
        phone_ids=phone_ids,
        phone_starts=phone_starts,
        phone_ends=phone_ends,
        phone_lens=phone_lens,
        utt_ids=utt_ids,
    )