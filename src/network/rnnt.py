# network/rnnt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TimeSubsample(nn.Module):
    """Simple frame-rate reduction by striding on time axis."""
    def __init__(self, factor: int = 4):
        super().__init__()
        assert factor >= 1
        self.factor = factor

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.factor == 1:
            return x, x_lens
        x = x[:, :: self.factor, :].contiguous()
        x_lens = (x_lens + (self.factor - 1)) // self.factor
        return x, x_lens


class RNNTEncoder(nn.Module):
    """
    Encoder: (B,T,F) -> (B,T',H)
    Uses (Bi)LSTM with time subsampling before LSTM to control memory.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        layers: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.1,
        subsample: int = 4,
    ) -> None:
        super().__init__()
        self.sub = TimeSubsample(subsample)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.sub(x, x_lens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out_lens = out_lens.to(out.device)
        return out, out_lens


class RNNTPredictor(nn.Module):
    """
    Predictor (a.k.a. prediction network):
      tokens -> (B,U,H)
    We use embedding + LSTM. Also provides a step() for greedy decoding.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden: int = 512,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.out_dim = hidden

    def forward(self, y_in: torch.Tensor, y_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_in: (B, U)  (already includes initial blank)
        emb = self.embed(y_in)  # (B,U,E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, y_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out, out_lens

    def init_state(self, batch_size: int, device: torch.device):
        # h, c: (layers, B, hidden)
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        return (h, c)

    def step(self, token: torch.Tensor, state):
        """
        token: (B,) int64
        Returns: out (B, H), new_state
        """
        emb = self.embed(token).unsqueeze(1)  # (B,1,E)
        out, new_state = self.lstm(emb, state)  # out: (B,1,H)
        return out.squeeze(1), new_state


class RNNTJoiner(nn.Module):
    """
    Joiner: combines encoder and predictor features -> logits over vocab.
    """
    def __init__(self, enc_dim: int, pred_dim: int, joint_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.act = nn.Tanh()
        self.out = nn.Linear(joint_dim, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Training (full):
          enc:  (B, T, 1, Henc)
          pred: (B, 1, U, Hpred)
          -> logits: (B, T, U, V)
        Greedy step:
          enc:  (B, Henc)
          pred: (B, Hpred)
          -> logits: (B, V)
        """
        if enc.dim() == 2:
            z = self.enc_proj(enc) + self.pred_proj(pred)
            z = self.act(z)
            return self.out(z)

        z = self.enc_proj(enc) + self.pred_proj(pred)
        z = self.act(z)
        return self.out(z)


class RNNTModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        blank_idx: int = 0,
        enc_hidden: int = 512,
        enc_layers: int = 3,
        enc_bidir: bool = True,
        enc_dropout: float = 0.1,
        enc_subsample: int = 4,
        pred_embed: int = 256,
        pred_hidden: int = 512,
        pred_layers: int = 2,
        pred_dropout: float = 0.1,
        joint_dim: int = 512,
    ) -> None:
        super().__init__()
        self.blank_idx = blank_idx
        self.encoder = RNNTEncoder(
            in_dim=in_dim,
            hidden=enc_hidden,
            layers=enc_layers,
            bidirectional=enc_bidir,
            dropout=enc_dropout,
            subsample=enc_subsample,
        )
        self.predictor = RNNTPredictor(
            vocab_size=vocab_size,
            embed_dim=pred_embed,
            hidden=pred_hidden,
            layers=pred_layers,
            dropout=pred_dropout,
        )
        self.joiner = RNNTJoiner(
            enc_dim=self.encoder.out_dim,
            pred_dim=self.predictor.out_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        feats: torch.Tensor,          # (B,T,F)
        feat_lens: torch.Tensor,      # (B,)
        targets: torch.Tensor,        # (B,Umax) padded, WITHOUT initial blank
        target_lens: torch.Tensor,    # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          log_probs: (B, T', Umax+1, V)  (log-softmax)
          out_lens: (B,)
          y_in_lens: (B,) = target_lens + 1
        """
        enc, out_lens = self.encoder(feats, feat_lens)  # (B,T',Henc)

        # prepend blank as y_in[0]
        B, Umax = targets.shape
        blank_col = torch.full((B, 1), self.blank_idx, dtype=torch.long, device=targets.device)
        y_in = torch.cat([blank_col, targets], dim=1)  # (B,Umax+1)
        y_in_lens = target_lens + 1

        pred, _ = self.predictor(y_in, y_in_lens)  # (B,Umax+1,Hpred)

        # joint full tensor (B,T',Umax+1,V)
        enc_ex = enc.unsqueeze(2)      # (B,T',1,Henc)
        pred_ex = pred.unsqueeze(1)    # (B,1,Umax+1,Hpred)
        logits = self.joiner(enc_ex, pred_ex)

        return logits, out_lens, y_in_lens
