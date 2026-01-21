# network/lstm_ctc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class LSTMCTC(nn.Module):
    """
    (Bi)LSTM acoustic model for CTC.
    Input:  (B, T, F)
    Output: log_probs (T, B, V)  for torch.nn.CTCLoss
    """
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        hidden: int = 512,
        layers: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.layers = layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, vocab_size)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F)
        x_lens: (B,) lengths in frames
        """
        # pack -> lstm -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.proj(out)                 # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)   # (T, B, V) for CTC
        return log_probs, out_lens
