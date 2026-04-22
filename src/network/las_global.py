# network/las_global.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _stack_subsample(
    x: torch.Tensor, lens: torch.Tensor, factor: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Causal stacking subsample:
      x: (B,T,F) -> (B, ceil(T/factor), F*factor)
      lens: (B,) -> ceil(lens/factor)
    """
    if factor <= 1:
        return x, lens
    B, T, F = x.shape
    pad = (-T) % factor
    if pad:
        x = torch.cat([x, x.new_zeros(B, pad, F)], dim=1)
        T = T + pad
    T2 = T // factor
    x = x.view(B, T2, factor * F)
    lens2 = (lens + factor - 1) // factor
    return x, lens2


class AdditiveAttention(nn.Module):
    """
    Bahdanau/additive attention with masking.
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.w_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        enc_out: torch.Tensor,   # (B,Te,EncDim)
        enc_lens: torch.Tensor,  # (B,)
        dec_state: torch.Tensor  # (B,DecDim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Te, _ = enc_out.shape
        # precompute W_h(enc)
        h = self.w_h(enc_out)                      # (B,Te,A)
        s = self.w_s(dec_state).unsqueeze(1)       # (B,1,A)
        e = self.v(torch.tanh(h + s)).squeeze(-1)  # (B,Te)

        # mask
        idx = torch.arange(Te, device=enc_lens.device).unsqueeze(0)  # (1,Te)
        mask = idx < enc_lens.unsqueeze(1)                           # (B,Te)
        neg_inf = torch.finfo(e.dtype).min         # <--关键：float16/float32都安全
        
        e = e.masked_fill(~mask, neg_inf)

        a = torch.softmax(e, dim=-1)                                 # (B,Te)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)          # (B,EncDim)
        return ctx, a


class LASGlobal(nn.Module):
    """
    Classic LAS-style Seq2Seq:
      - Encoder: (Bi)LSTM on acoustic features (with stacking subsample)
      - Attention: global additive attention
      - Decoder: LSTM with input-feeding (concat embedding + prev context)
      - Output: vocab logits per step
    """
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        enc_hidden: int,
        enc_layers: int,
        enc_bidirectional: bool,
        dec_embed: int,
        dec_hidden: int,
        dec_layers: int,
        attn_dim: int,
        dropout: float = 0.1,
        enc_subsample: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.enc_subsample = int(enc_subsample)

        enc_dir = 2 if enc_bidirectional else 1
        self.enc = nn.LSTM(
            input_size=in_dim * self.enc_subsample,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            batch_first=True,
            bidirectional=enc_bidirectional,
            dropout=(dropout if enc_layers > 1 else 0.0),
        )
        self.enc_out_dim = enc_hidden * enc_dir

        self.emb = nn.Embedding(vocab_size, dec_embed)

        # input-feeding: [emb, prev_ctx] -> decoder LSTM
        self.dec = nn.LSTM(
            input_size=dec_embed + self.enc_out_dim,
            hidden_size=dec_hidden,
            num_layers=dec_layers,
            batch_first=True,
            dropout=(dropout if dec_layers > 1 else 0.0),
        )

        self.attn = AdditiveAttention(enc_dim=self.enc_out_dim, dec_dim=dec_hidden, attn_dim=attn_dim)

        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dec_hidden + self.enc_out_dim, vocab_size),
        )

    def forward(
        self,
        feats: torch.Tensor,      # (B,T,F)
        feat_lens: torch.Tensor,  # (B,)
        y_in: torch.Tensor,       # (B,U) teacher forcing inputs (BOS, w1, w2, ...)
    ) -> torch.Tensor:
        device = feats.device
        feat_lens = feat_lens.to(device)

        # --- subsample by stacking ---
        x, lens = _stack_subsample(feats, feat_lens, self.enc_subsample)  # (B,Te,F*s), (B,)

        # --- encoder with packing ---
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.enc(packed)
        enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)    # (B,Te,EncDim)

        B, U = y_in.shape
        emb = self.emb(y_in)                                             # (B,U,E)

        # decoding loop (input-feeding)
        ctx_prev = enc_out.new_zeros(B, self.enc_out_dim)                # (B,EncDim)
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        logits = []
        for t in range(U):
            dec_in_t = torch.cat([emb[:, t, :], ctx_prev], dim=-1).unsqueeze(1)  # (B,1,E+EncDim)
            dec_out_t, state = self.dec(dec_in_t, state)                          # (B,1,H)
            dec_h = dec_out_t.squeeze(1)                                          # (B,H)
            ctx, _ = self.attn(enc_out=enc_out, enc_lens=lens, dec_state=dec_h)   # (B,EncDim)
            y = self.out(torch.cat([dec_h, ctx], dim=-1))                         # (B,V)
            logits.append(y.unsqueeze(1))
            ctx_prev = ctx

        return torch.cat(logits, dim=1)  # (B,U,V)
