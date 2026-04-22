# utils/srv.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import torch


SrvValue = Literal["pm1", "binary"]


def _sample_unique_indices(dim: int, k: int, g: torch.Generator) -> torch.Tensor:
    """
    Sample k unique indices in [0, dim).
    k << dim, so simple resample-until-unique is fine.
    """
    idx = torch.randint(0, dim, (k,), generator=g)
    idx = torch.unique(idx)
    while idx.numel() < k:
        extra = torch.randint(0, dim, (k - idx.numel(),), generator=g)
        idx = torch.unique(torch.cat([idx, extra], dim=0))
    return idx[:k]


def make_srv_table(
    vocab_size: int,
    dim: int,
    k: int,
    seed: int,
    value: SrvValue = "pm1",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns dense SRV table: (V, D) with exactly k non-zeros per row.
    Rows are L2-normalized (unit norm) for cosine similarity.
    """
    assert vocab_size > 0
    assert 1 <= k <= dim

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    table = torch.zeros(vocab_size, dim, dtype=dtype)

    for wid in range(vocab_size):
        pos = _sample_unique_indices(dim, k, g)

        if value == "binary":
            vals = torch.ones(k, dtype=dtype)
        elif value == "pm1":
            # random +/-1 with equal prob
            r = torch.rand(k, generator=g)
            vals = torch.where(r < 0.5, torch.tensor(-1.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
        else:
            raise ValueError(f"Unknown SRV value type: {value}")

        table[wid, pos] = vals

    # L2 normalize each row (avoid div0)
    norm = torch.linalg.norm(table, dim=1, keepdim=True).clamp_min(1e-8)
    table = table / norm
    return table
