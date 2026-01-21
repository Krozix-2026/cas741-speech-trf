import torch
import torch.nn as nn
import torch.optim as optim



class LocalistRNN(nn.Module):
    def __init__(self, input_dim=64, hidden=2048, layers=1, out_dim=900, dropout=0.9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.readout = nn.Linear(hidden, out_dim)
        self._init_forget_bias(1.0)

    def _init_forget_bias(self, val=1.0):
        for name, p in self.lstm.named_parameters():
            if "bias" in name:
                H = p.shape[0] // 4
                with torch.no_grad():
                    p[H:2*H].fill_(val)

    def forward(self, x, h=None, return_sequence: bool = False):
        """
        x: (B, T, input_dim)
        h: initial hidden state (optional)
        return_sequence:
          - False: 保持原行为 -> 返回 logits, (h_n, c_n)
          - True:  返回 logits, o (整条时间上的 hidden 序列)
        """
        o, h_last = self.lstm(x, h)      # o: (B, T, hidden)
        logits = self.readout(o)         # (B, T, out_dim)

        if return_sequence:
            return logits, o, h_last            # o: (B, T, hidden)
        else:
            return logits, h_last        # (h_n, c_n)



