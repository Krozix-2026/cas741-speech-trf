# network/lstm_frame_srv_semantic.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100

class LSTMFrameSRVSemantic(nn.Module):
    """
    Frame LSTM (SRV per frame) + Word-level LSTM semantic head (predict next-word SRV at word boundaries).
    """
    def __init__(
        self,
        in_dim: int,
        srv_dim: int,
        frame_hidden: int = 512,
        frame_layers: int = 3,
        dropout: float = 0.1,

        word_rep_dim: int = 256,
        word_lstm_hidden: int = 256,
        word_lstm_layers: int = 1,
        rep_dropout: float = 0.1,
        word_dropout_p: float = 0.25,
        rep_noise_std: float = 0.0,
    ) -> None:
        super().__init__()

        self.frame_lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=frame_hidden,
            num_layers=frame_layers,
            dropout=dropout if frame_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.frame_proj = nn.Linear(frame_hidden, srv_dim)

        # bottleneck (anti "word-id lookup")!!
        self.word_proj = nn.Sequential(
            nn.Linear(frame_hidden, word_rep_dim),
            nn.LayerNorm(word_rep_dim),
            nn.GELU(),
            nn.Dropout(rep_dropout),
        )

        self.word_dropout_p = float(word_dropout_p)
        self.rep_noise_std = float(rep_noise_std)

        self.word_lstm = nn.LSTM(
            input_size=word_rep_dim,
            hidden_size=word_lstm_hidden,
            num_layers=word_lstm_layers,
            dropout=dropout if word_lstm_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )

        self.sem_proj = nn.Linear(word_lstm_hidden, srv_dim)

    def forward_with_hidden(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # 1. frame LSTM
        packed_out, _ = self.frame_lstm(packed)
        
        h, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)
        
        y = self.frame_proj(h) # FCN
        
        return y, h, out_lens

    def _extract_word_reps(
        self,
        h: torch.Tensor, # (B,T,H)
        out_lens: torch.Tensor, # (B,)
        word_starts: torch.Tensor, # (B,W)
        word_ends: torch.Tensor, # (B,W)
        word_lens: torch.Tensor, # (B,)
        tail_frames: int,
    ) -> torch.Tensor:
        """
        r_i = mean_{t in [max(ef-tail_frames,sf), ef)} h_t
        returns reps: (B,W,H)
        """
        B, T, Hdim = h.shape
        W = int(word_ends.shape[1])
        reps = h.new_zeros((B, W, Hdim))

        for b in range(B):
            Tb = int(out_lens[b].item())
            wb = int(word_lens[b].item())
            if wb <= 0:
                continue
            for j in range(wb):
                sf = int(word_starts[b, j].item())
                ef = int(word_ends[b, j].item())
                if ef <= 0:
                    continue
                sf = max(sf, 0)
                ef = min(ef, Tb)
                if ef <= sf:
                    continue
                i0 = max(ef - int(tail_frames), sf)
                if ef <= i0:
                    continue
                reps[b, j] = h[b, i0:ef].mean(dim=0)
        return reps

    def _apply_word_dropout(self, r: torch.Tensor, word_lens: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.word_dropout_p <= 0:
            return r
        B, W, D = r.shape
        # valid positions mask
        pos = torch.arange(W, device=r.device).unsqueeze(0).expand(B, W)
        valid = pos < word_lens.unsqueeze(1)  # (B,W)
        drop = (torch.rand(B, W, device=r.device) < self.word_dropout_p) & valid
        r = r.masked_fill(drop.unsqueeze(-1), 0.0)
        return r

    def forward(
        self,
        x: torch.Tensor, # (B,T,F)
        x_lens: torch.Tensor, # (B,)
        word_starts: torch.Tensor, # (B,W)
        word_ends: torch.Tensor, # (B,W)
        word_lens: torch.Tensor, # (B,)
        tail_frames: int,
        detach_frame_for_sem: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          y_frame: (B,T, srv_dim)  for word-tail SRV loss
          out_lens: (B,)
          pred_sem: (B,W, srv_dim) semantic predictions aligned to each word position i (predict word i+shift in trainer)
        """
        # print("x:", x.shape)#[32, 1698, 64] (B,T,F)
        # print("x_lens:", x_lens)# (B,)
        y_frame, h, out_lens = self.forward_with_hidden(x, x_lens)
        # print("y_frame:", y_frame.shape)#[32, 1698, 2048]
        # print("h:", h.shape)#[32, 1698, 512]
        # print("out_lens:", out_lens.shape)#[32]

        # print("word_ends:", word_ends)#
        # print("word_ends:", word_ends.shape)#(B,W)
        W = int(word_ends.shape[1])
        # print("W:", W)#
        if W == 0:
            pred_sem = y_frame.new_zeros((x.size(0), 0, y_frame.size(-1)))
            return y_frame, out_lens, pred_sem

        reps_h = self._extract_word_reps(h, out_lens, word_starts, word_ends, word_lens, tail_frames)
        # print("reps_h:", reps_h)#[32, 49, 512]
        if detach_frame_for_sem:
            reps_h = reps_h.detach()

        r = self.word_proj(reps_h)  # (B,W,word_rep_dim)
        # print("r:", r.shape)#[32, 49, 256]

        if self.rep_noise_std > 0 and self.training:
            r = r + torch.randn_like(r) * self.rep_noise_std

        r = self._apply_word_dropout(r, word_lens)

        # pack word sequences (need len>=1)
        lens = word_lens.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            r, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # 2: word LSTM
        packed_out, _ = self.word_lstm(packed)

        c, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=W)  # (B,W,Hw)
        # print("c:", c.shape)#[32, 49, 256]
        
        pred_sem = self.sem_proj(c)  # (B,W,srv_dim)
        # print("pred_sem:", pred_sem.shape)#[32, 49, 2048]
        
        return y_frame, out_lens, pred_sem