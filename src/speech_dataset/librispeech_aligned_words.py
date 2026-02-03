# speech_dataset/librispeech_aligned_words.py
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
class WordVocab:
    stoi: Dict[str, int]
    itos: List[str]
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.itos)

def build_word_vocab_from_manifest(manifest_path: Path, topk: int = 20000) -> WordVocab:
    from collections import Counter
    cnt = Counter()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            for w, sf, ef in obj["words"]:
                cnt[w] += 1
    # reserve ids: 0=<unk>
    itos = ["<unk>"] + [w for w, _ in cnt.most_common(topk)]
    stoi = {w: i for i, w in enumerate(itos)}
    return WordVocab(stoi=stoi, itos=itos, unk_id=0)

class LibriSpeechAlignedWords(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        subset: str,
        vocab: WordVocab,
        tail_frames: int = 10,
        limit_samples: Optional[int] = None,
    ) -> None:
        self.vocab = vocab
        self.tail_frames = int(tail_frames)

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.items[idx]
        coch = np.load(obj["coch_path"])  # (64, T) float16
        if coch.ndim != 2 or coch.shape[0] != 64:
            raise ValueError(f"Unexpected coch shape {coch.shape} for {obj['coch_path']}")
        # -> (T, 64)
        feat = torch.from_numpy(coch.astype("float32")).transpose(0, 1).contiguous()
        T = feat.shape[0]

        target = torch.full((T,), IGNORE, dtype=torch.long)

        for w, sf, ef in obj["words"]:
            sf = int(sf); ef = int(ef)
            if ef <= 0 or sf >= T:
                continue
            sf = max(sf, 0)
            ef = min(ef, T)
            if ef <= sf:
                continue
            wid = self.vocab.stoi.get(w, self.vocab.unk_id)

            # supervise tail frames
            i0 = max(ef - self.tail_frames, sf)
            if ef > i0:
                target[i0:ef] = wid

        return {
            "feat": feat,              # (T, 64)
            "feat_len": T,
            "target": target,          # (T,)
            "utt_id": obj["utt_id"],
        }