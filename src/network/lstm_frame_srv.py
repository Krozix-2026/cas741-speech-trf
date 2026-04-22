# network/lstm_frame_srv.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class LSTMFrameSRV(nn.Module):
    """
    Unidirectional LSTM that outputs SRV vectors per frame.
    Input:  (B, T, F)
    Output: y (B, T, D), out_lens (B,)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,          # SRV dim D
        hidden: int = 512,
        layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise ValueError("This SRV model is configured to be unidirectional (bidirectional=False).")

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        y = self.proj(out)  # (B, T, D)
        return y, out_lens

    def forward_with_hidden(self, x: torch.Tensor, x_lens: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)
        y = self.proj(out)  # (B,T,D)
        return y, out, out_lens

if __name__ == "__main__":
    in_dim = 64
    out_dim = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 准备模拟输入数据
    # 假设 Batch Size 为 2，最大长度 T 为 120
    B, T = 2, 120
    x = torch.randn(B, T, in_dim).to(device)
    
    # 模拟实际长度（例如第一个样本长120，第二个样本长100）
    x_lens = torch.tensor([120, 100], dtype=torch.long) 

    # 3. 实例化模型并移动到设备
    lstm = LSTMFrameSRV(in_dim=in_dim, out_dim=out_dim).to(device)
    
    # 4. 前向传播
    # 注意：根据你的代码逻辑，x_lens 不需要手动传到 GPU，
    # 因为内部调用 pack_padded_sequence 时用了 .cpu()
    y, out_lens = lstm(x, x_lens)

    # 5. 打印结果确认
    print(f"Input shape:  {x.shape}")      # [2, 120, 64]
    print(f"Output shape: {y.shape}")      # [2, 120, 1024]
    print(f"Output lens:  {out_lens}")     # tensor([120, 100])
    
    
    
    
