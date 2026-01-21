# train/eva.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from speech_dataset.vocab import CharVocab, normalize_text


def _ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int) -> List[List[int]]:
    """
    log_probs: (T, B, V)
    return: list of decoded id sequences (collapsed, removed blank)
    """
    pred = torch.argmax(log_probs, dim=-1)  # (T, B)
    pred = pred.transpose(0, 1)             # (B, T)

    outs: List[List[int]] = []
    for b in range(pred.size(0)):
        seq = pred[b].tolist()
        collapsed: List[int] = []
        prev = None
        for x in seq:
            if x == prev:
                continue
            prev = x
            if x != blank_idx:
                collapsed.append(x)
        outs.append(collapsed)
    return outs


def _edit_distance(a: List[str], b: List[str]) -> int:
    # classic DP
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    r = ref.split() if ref else []
    h = hyp.split() if hyp else []
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / float(len(r))


def cer(ref: str, hyp: str) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    r = list(ref)
    h = list(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / float(len(r))


@dataclass
class EvalResult:
    mean_loss: float
    wer: float
    cer: float


@torch.no_grad()
def evaluate_batch(
    log_probs: torch.Tensor,         # (T, B, V)
    input_lens: torch.Tensor,        # (B,)
    texts: List[str],
    vocab: CharVocab,
) -> Tuple[List[str], List[float], List[float]]:
    decoded_ids = _ctc_greedy_decode(log_probs, blank_idx=vocab.blank_idx)

    hyps: List[str] = []
    wers: List[float] = []
    cers: List[float] = []

    for hyp_ids, ref in zip(decoded_ids, texts):
        hyp = vocab.decode_ids(hyp_ids)
        ref_n = normalize_text(ref)
        hyps.append(hyp)
        wers.append(wer(ref_n, hyp))
        cers.append(cer(ref_n, hyp))

    return hyps, wers, cers
