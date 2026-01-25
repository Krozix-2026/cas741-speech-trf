# speech_dataset/librispeech.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset

from speech_dataset.vocab import CharVocab, normalize_text


@dataclass
class LibriSpeechBatch:
    feats: torch.Tensor            # (B, T, F)
    feat_lens: torch.Tensor        # (B,)
    targets: torch.Tensor          # (sumU,)
    target_lens: torch.Tensor      # (B,)
    texts: List[str]               # normalized transcript
    utt_ids: List[str]             # "spk-chap-utt"


class LibriSpeechASR(Dataset):
    """
    Wrap torchaudio.datasets.LIBRISPEECH but DO NOT use ds[i] to load audio
    (to avoid torchcodec backend). Instead:
      - ds.get_metadata(i) -> rel_path, sr, transcript, spk, chap, utt
      - soundfile reads FLAC from disk
      - then compute log-mel features
    """
    def __init__(
        self,
        root: Path,
        subset: str,
        vocab: CharVocab,
        sample_rate: int = 16000,
        n_mels: int = 80,
        win_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        download: bool = True,
        limit_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.subset = subset
        self.vocab = vocab
        self.sample_rate = sample_rate

        self.ds = torchaudio.datasets.LIBRISPEECH(
            root=str(self.root),
            url=subset,
            download=download,
        )

        # torchaudio LibriSpeech 默认解压到 root/LibriSpeech/...
        self.archive_root = self.root / "LibriSpeech"

        if limit_samples is not None:
            limit_samples = int(limit_samples)
            self.indices = list(range(min(limit_samples, len(self.ds))))
        else:
            self.indices = None

        win_length = int(sample_rate * (win_length_ms / 1000.0))
        hop_length = int(sample_rate * (hop_length_ms / 1000.0))

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.ds)

    def _load_audio_soundfile(self, abs_path: Path) -> tuple[torch.Tensor, int]:
        audio, sr = sf.read(str(abs_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:  # (T, C) -> mono
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(np.asarray(audio)).unsqueeze(0)  # (1, T)
        return wav, sr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = self.indices[idx] if self.indices is not None else idx

        #不触发 torchaudio.load
        rel_path, sr, transcript, spk_id, chap_id, utt_id = self.ds.get_metadata(real_idx)
        abs_path = self.archive_root / rel_path

        waveform, sr2 = self._load_audio_soundfile(abs_path)

        # resample to target sample_rate if needed
        if sr2 != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr2, self.sample_rate)

        # waveform: (1, N) -> mel: (1, n_mels, T) -> (T, n_mels)
        with torch.no_grad():
            mel = self.melspec(waveform)                 # (1, n_mels, T)
            mel = self.to_db(mel)                        # (1, n_mels, T)
            mel = mel.squeeze(0).transpose(0, 1).contiguous()  # (T, n_mels)

        text = normalize_text(transcript)
        y = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        uid = f"{spk_id}-{chap_id}-{utt_id}"
        return {
            "feat": mel,
            "feat_len": mel.shape[0],
            "target": y,
            "target_len": y.numel(),
            "text": text,
            "utt_id": uid,
        }


def collate_librispeech(items: List[Dict[str, Any]]) -> LibriSpeechBatch:
    items = sorted(items, key=lambda x: x["feat_len"], reverse=True)

    feat_lens = torch.tensor([it["feat_len"] for it in items], dtype=torch.long)
    target_lens = torch.tensor([it["target_len"] for it in items], dtype=torch.long)

    max_t = int(feat_lens.max().item())
    feat_dim = int(items[0]["feat"].shape[1])

    feats = torch.zeros(len(items), max_t, feat_dim, dtype=torch.float32)
    for i, it in enumerate(items):
        t = it["feat_len"]
        feats[i, :t] = it["feat"]

    targets = torch.cat([it["target"] for it in items], dim=0)

    texts = [it["text"] for it in items]
    utt_ids = [it["utt_id"] for it in items]

    return LibriSpeechBatch(
        feats=feats,
        feat_lens=feat_lens,
        targets=targets,
        target_lens=target_lens,
        texts=texts,
        utt_ids=utt_ids,
    )

def collate_rnnt(items: List[Dict[str, Any]]):
    """
    For RNNT:
      feats: (B, T, F) padded
      targets: (B, U) padded (NO concatenation)
    """
    items = sorted(items, key=lambda x: x["feat_len"], reverse=True)

    feat_lens = torch.tensor([it["feat_len"] for it in items], dtype=torch.long)
    target_lens = torch.tensor([it["target_len"] for it in items], dtype=torch.long)

    max_t = int(feat_lens.max().item())
    feat_dim = int(items[0]["feat"].shape[1])
    feats = torch.zeros(len(items), max_t, feat_dim, dtype=torch.float32)
    for i, it in enumerate(items):
        t = it["feat_len"]
        feats[i, :t] = it["feat"]

    max_u = int(target_lens.max().item()) if len(items) > 0 else 0
    targets = torch.full((len(items), max_u), 0, dtype=torch.long)  # pad with blank(0)
    for i, it in enumerate(items):
        u = it["target_len"]
        if u > 0:
            targets[i, :u] = it["target"]

    texts = [it["text"] for it in items]
    utt_ids = [it["utt_id"] for it in items]

    return {
        "feats": feats,
        "feat_lens": feat_lens,
        "targets": targets,
        "target_lens": target_lens,
        "texts": texts,
        "utt_ids": utt_ids,
    }

