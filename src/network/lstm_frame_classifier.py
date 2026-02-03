# network/lstm_frame_classifier.py
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class LSTMFrameClassifier(nn.Module):
    """
    Unidirectional LSTM frame classifier.
    Input:  (B, T, F)
    Output: logits (B, T, V), out_lens (B,)
    """
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        hidden: int = 512,
        layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,  # <-- default OFF (no bi-LSTM)
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("This classifier is configured to be unidirectional (bidirectional=False).")

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
            bidirectional=False,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F)
        x_lens: (B,) lengths in frames
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.proj(out)  # (B, T, V) raw logits
        return logits, out_lens
    
    def forward_with_hidden(self, x, x_lens):
        packed = nn.utils.rnn.pack_padded_sequence( x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # out:(B,T,H)
        logits = self.proj(out)  # (B,T,V)
        return logits, out, out_lens