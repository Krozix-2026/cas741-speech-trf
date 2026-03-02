# network/las_monotonic.py
from __future__ import annotations

from typing import Tuple, List
import torch
import torch.nn as nn


class MonotonicAttention(nn.Module):
    """
    Differentiable expected monotonic attention (Raffel et al., 2017 style).
    This enforces monotonic alignments (no attending to "earlier than previous step").
    """
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.eps = 1e-6

    def forward(
        self,
        enc: torch.Tensor,         # (B, T, H)
        enc_lens: torch.Tensor,    # (B,)
        dec_query: torch.Tensor,   # (B, D) (usually decoder hidden)
        alpha_prev: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H = enc.shape
        device = enc.device

        # ---- force fp32 inside attention for numerical stability under AMP ----
        enc_f = enc.float()
        dec_q_f = dec_query.float()
        alpha_prev_f = alpha_prev.float()
        enc_lens = enc_lens.to(device)

        # energy: (B,T) in fp32
        enc_proj = self.W_h(enc_f)                          # (B,T,A)
        dec_proj = self.W_s(dec_q_f).unsqueeze(1)           # (B,1,A)
        e = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B,T)

        # mask padding => very negative (safe in fp32)
        t_idx = torch.arange(T, device=device).unsqueeze(0)       # (1,T)
        pad_mask = t_idx >= enc_lens.unsqueeze(1)                 # (B,T)
        e = e.masked_fill(pad_mask, torch.finfo(e.dtype).min)

        p = torch.sigmoid(e)                                      # (B,T) fp32

        # ---- NO in-place recurrence: build alpha columns in a list ----
        alpha_cols = []

        # j=0
        alpha0 = p[:, 0] * alpha_prev_f[:, 0]                     # (B,)
        alpha_cols.append(alpha0)

        for j in range(1, T):
            prev = alpha_cols[-1]                                 # alpha_{j-1} (B,)
            term = alpha_prev_f[:, j] + prev * (1.0 - p[:, j - 1]) / (p[:, j - 1] + self.eps)
            alpha_j = p[:, j] * term                               # (B,)
            alpha_cols.append(alpha_j)

        alpha = torch.stack(alpha_cols, dim=1)                    # (B,T) fp32

        # context: (B,H) computed in fp32, then cast back to enc dtype
        context = torch.bmm(alpha.unsqueeze(1), enc_f).squeeze(1)  # fp32
        context = context.to(enc.dtype)                            # back to fp16/bf16 if needed

        # keep alpha fp32 for stability across decoder steps
        return context, alpha


class CausalLASMonotonic(nn.Module):
    """
    Strictly causal LAS-like seq2seq:
      - Encoder: UniLSTM over (possibly subsampled) gammatone frames
      - Attention: expected monotonic attention
      - Decoder: LSTM (autoregressive LM) with teacher forcing

    Forward takes y_in (B,U) and returns logits (B,U,V).
    """
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        enc_hidden: int = 512,
        enc_layers: int = 3,
        dec_embed: int = 256,
        dec_hidden: int = 512,
        dec_layers: int = 2,
        attn_dim: int = 128,
        dropout: float = 0.1,
        enc_subsample: int = 2,
    ) -> None:
        super().__init__()
        assert enc_subsample in (1, 2, 4), "enc_subsample should be 1/2/4 for first version"

        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.enc_hidden = enc_hidden
        self.enc_layers = enc_layers
        self.dec_embed = dec_embed
        self.dec_hidden = dec_hidden
        self.dec_layers = dec_layers
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.enc_subsample = enc_subsample

        # causal subsample by stacking frames (B,T,F) -> (B,T',F*s)
        enc_in_dim = in_dim * enc_subsample
        self.enc_in = nn.Linear(enc_in_dim, enc_hidden)

        self.encoder = nn.LSTM(
            input_size=enc_hidden,
            hidden_size=enc_hidden,
            num_layers=enc_layers,
            dropout=dropout if enc_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )

        self.attn = MonotonicAttention(enc_dim=enc_hidden, dec_dim=dec_hidden, attn_dim=attn_dim)

        self.emb = nn.Embedding(vocab_size, dec_embed)

        # stacked LSTMCells for step-wise decoding
        self.dec_cells = nn.ModuleList()
        self.dec_drop = nn.Dropout(dropout)

        # first layer input: [token_emb ; context]
        self.dec_cells.append(nn.LSTMCell(dec_embed + enc_hidden, dec_hidden))
        for _ in range(1, dec_layers):
            self.dec_cells.append(nn.LSTMCell(dec_hidden, dec_hidden))

        self.out_proj = nn.Linear(dec_hidden + enc_hidden, vocab_size)

    def _subsample(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal subsample by stacking consecutive frames.
        x: (B,T,F) -> (B,T',F*s)
        """
        if self.enc_subsample == 1:
            return x, x_lens

        B, T, F = x.shape
        s = self.enc_subsample
        T_trim = (T // s) * s
        x = x[:, :T_trim, :]
        x = x.reshape(B, T_trim // s, F * s)
        x_lens = (x_lens // s).clamp_min(1)
        return x, x_lens

    def encode(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self._subsample(x, x_lens)
        x = self.enc_in(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.encoder(packed)
        enc_out, enc_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        enc_lens = enc_lens.to(enc_out.device)  # <-- 关键：把长度张量搬到同设备
        return enc_out, enc_lens

    def forward(
        self,
        x: torch.Tensor,          # (B,T,F)
        x_lens: torch.Tensor,     # (B,)
        y_in: torch.Tensor,       # (B,U) teacher-forcing inputs (BOS + tokens + PAD)
    ) -> torch.Tensor:
        """
        Returns:
          logits: (B,U,V)
        """
        device = x.device
        enc, enc_lens = self.encode(x, x_lens)     # (B,T',H)
        B, T, H = enc.shape
        U = y_in.size(1)

        # init decoder states
        h_list: List[torch.Tensor] = []
        c_list: List[torch.Tensor] = []
        for _ in range(self.dec_layers):
            h_list.append(torch.zeros(B, self.dec_hidden, device=device))
            c_list.append(torch.zeros(B, self.dec_hidden, device=device))

        # init alpha_prev: attend starts at position 0
        alpha_prev = torch.zeros(B, T, device=device)
        alpha_prev[:, 0] = 1.0

        logits_steps: List[torch.Tensor] = []

        for u in range(U):
            token_u = y_in[:, u]                 # (B,)
            emb_u = self.emb(token_u)            # (B,E)

            # use top-layer decoder hidden as query for attention
            dec_query = h_list[-1]               # (B,D)
            context, alpha_prev = self.attn(enc, enc_lens, dec_query, alpha_prev)  # (B,H), (B,T)

            dec_in = torch.cat([emb_u, context], dim=-1)  # (B, E+H)

            # stacked LSTMCell
            for li, cell in enumerate(self.dec_cells):
                h, c = cell(dec_in, (h_list[li], c_list[li]))
                h_list[li], c_list[li] = h, c
                dec_in = self.dec_drop(h) if li < (self.dec_layers - 1) else h

            top_h = h_list[-1]
            out = torch.cat([top_h, context], dim=-1)
            logits_u = self.out_proj(out)        # (B,V)
            logits_steps.append(logits_u)

        logits = torch.stack(logits_steps, dim=1)  # (B,U,V)
        return logits
