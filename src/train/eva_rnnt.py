# train/eva_rnnt.py
from __future__ import annotations

from typing import List, Tuple

import torch

from speech_dataset.vocab import CharVocab, normalize_text


def _edit_distance(a: List[str], b: List[str]) -> int:
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


@torch.no_grad()
def rnnt_greedy_decode(
    model,
    feats: torch.Tensor,      # (B,T,F)
    feat_lens: torch.Tensor,  # (B,)
    vocab: CharVocab,
    max_symbols_per_step: int = 5,
) -> List[str]:
    """
    Simple RNNT greedy decoding (slow but works for sanity check).
    """
    device = feats.device
    model.eval()
    enc, out_lens = model.encoder(feats, feat_lens)  # (B,T',Henc)

    B = enc.size(0)
    hyps: List[str] = []
    for b in range(B):
        T = int(out_lens[b].item())
        enc_b = enc[b, :T]  # (T,Henc)

        state = model.predictor.init_state(1, device=device)
        last = torch.tensor([model.blank_idx], device=device, dtype=torch.long)

        out_tokens: List[int] = []
        for t in range(T):
            symbols = 0
            while symbols < max_symbols_per_step:
                pred_vec, state = model.predictor.step(last, state)  # (1,Hpred)
                logits = model.joiner(enc_b[t].unsqueeze(0), pred_vec)  # (1,V)
                k = int(torch.argmax(logits, dim=-1).item())
                if k == model.blank_idx:
                    break
                out_tokens.append(k)
                last = torch.tensor([k], device=device, dtype=torch.long)
                symbols += 1

        hyp = vocab.decode_ids(out_tokens)
        hyps.append(hyp)

    return hyps


@torch.no_grad()
def eval_metrics(texts: List[str], hyps: List[str]) -> Tuple[float, float]:
    wers = [wer(r, h) for r, h in zip(texts, hyps)]
    cers = [cer(r, h) for r, h in zip(texts, hyps)]
    return float(sum(wers) / max(1, len(wers))), float(sum(cers) / max(1, len(cers)))
