# speech_dataset/librispeech_aligned_phonemes.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = -100


@dataclass
class PhoneVocab:
    stoi: Dict[str, int]
    itos: List[str]
    unk_id: int = 0

    @property
    def size(self) -> int:
        return len(self.itos)


def build_phone_vocab_from_manifest(manifest_path: Path, topk: Optional[int] = None) -> PhoneVocab:
    """
    Expect manifest has:
      phonemes: [[phone, sf, ef], ...]
    """
    from collections import Counter
    cnt = Counter()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for p, sf, ef in obj.get("phonemes", []):
                cnt[p] += 1

    phones = [p for p, _ in cnt.most_common(topk)] if topk else [p for p, _ in cnt.most_common()]
    itos = ["<unk>"] + phones
    stoi = {p: i for i, p in enumerate(itos)}
    return PhoneVocab(stoi=stoi, itos=itos, unk_id=0)


class LibriSpeechAlignedPhonemes(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        subset: str,
        vocab: PhoneVocab,
        label_region: str = "all",      # "all" | "center" | "tail"
        tail_frames: int = 3,           # used if label_region="tail"
        center_frames: int = 3,         # used if label_region="center"
        limit_samples: Optional[int] = None,
    ) -> None:
        print("manifest_path:", manifest_path)
        self.vocab = vocab
        self.label_region = str(label_region).lower()
        self.tail_frames = int(tail_frames)
        self.center_frames = int(center_frames)

        self.items = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj["subset"] != subset:
                    continue
                self.items.append(obj)

        if limit_samples is not None:
            self.items = self.items[: int(limit_samples)]

    def __len__(self) -> int:
        return len(self.items)

    def _apply_region(self, target: torch.Tensor, pid: int, sf: int, ef: int):
        if self.label_region == "all":
            target[sf:ef] = pid
        elif self.label_region == "tail":
            i0 = max(ef - self.tail_frames, sf)
            if ef > i0:
                target[i0:ef] = pid
        elif self.label_region == "center":
            # label a small chunk around the center to avoid boundary jitter
            mid = (sf + ef) // 2
            half = max(1, self.center_frames // 2)
            i0 = max(sf, mid - half)
            i1 = min(ef, i0 + self.center_frames)
            if i1 > i0:
                target[i0:i1] = pid
        else:
            raise ValueError(f"Unknown label_region={self.label_region} (all|tail|center)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.items[idx]
        coch = np.load(obj["coch_path"])  # (64, T)
        if coch.ndim != 2 or coch.shape[0] != 64:
            raise ValueError(f"Unexpected coch shape {coch.shape} for {obj['coch_path']}")

        feat = torch.from_numpy(coch.astype("float32")).transpose(0, 1).contiguous()  # (T,64)
        T = feat.shape[0]
        target = torch.full((T,), IGNORE, dtype=torch.long)

        phone_ids: List[int] = []
        phone_starts: List[int] = []
        phone_ends: List[int] = []

        for p, sf, ef in obj.get("phonemes", []):
            sf = int(sf); ef = int(ef)
            if ef <= 0 or sf >= T:
                continue
            sf = max(sf, 0)
            ef = min(ef, T)
            if ef <= sf:
                continue

            pid = self.vocab.stoi.get(p, self.vocab.unk_id)
            phone_ids.append(pid)
            phone_starts.append(sf)
            phone_ends.append(ef)

            self._apply_region(target, pid, sf, ef)

        return {
            "feat": feat,
            "feat_len": T,
            "target": target,
            "phone_ids": torch.tensor(phone_ids, dtype=torch.long),
            "phone_starts": torch.tensor(phone_starts, dtype=torch.long),
            "phone_ends": torch.tensor(phone_ends, dtype=torch.long),
            "utt_id": obj["utt_id"],
        }