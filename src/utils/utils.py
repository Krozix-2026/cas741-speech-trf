# utils.py
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import warnings
import pickle



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logger(log_dir: Path, name: str = "run") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def _json_default(o: Any):
    # pathlib
    if isinstance(o, Path):
        return str(o)
    # numpy scalars/arrays
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    # torch scalars/tensors
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    # fallback
    return str(o)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.cnt += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def compress_and_norm_64T(G_64T: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    Input:  (64, T) non-negative
    Output: (T, 64) float32 ; log-compress + robust per-band [0,1]
    """
    G = np.maximum(G_64T, 0.0)
    Gc = np.log1p(k * G)
    lo = np.quantile(Gc, 0.01, axis=1, keepdims=True)
    hi = np.quantile(Gc, 0.99, axis=1, keepdims=True)
    Gn = (Gc - lo) / (hi - lo + 1e-6)
    Gn = np.clip(Gn, 0.0, 1.0).astype(np.float32)  # (64, T)
    return Gn.T  # -> (T, 64)


def load_pickle_quiet(path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        with open(path, "rb") as f:
            return pickle.load(f)

