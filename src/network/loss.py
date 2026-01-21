import torch
import torch.nn as nn
import torch.nn.functional as F


def cohort_weighted_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    target_multiplier: float = 1.0,
    non_target_multiplier: float = 1.0,
    target_embedded_region: bool = False,
    axis: int = -1,
    from_logits: bool = False,
) -> torch.Tensor:
    """
    Lenient penalty for non-target activations.

    y_pred: (batch, time, word)  —— 预测概率或logits
    y_true: 
      - 如果 target_embedded_region=True: (batch, time, word+1)
          最后一维的最后一个通道是 region mask (0/1)
      - 否则: (batch, time, word)
    """
    # 保证 dtype 一致
    y_true = y_true.to(dtype=y_pred.dtype)

    if target_embedded_region:
        # y_true[..., :-1] = one-hot targets
        # y_true[..., -1:] = region mask
        region = y_true[..., -1:]
        y_true = y_true[..., :-1]
    else:
        # 任意有 target 的位置都视作“相关 region”
        region = (y_true > 0).any(dim=-1, keepdim=True).to(y_pred.dtype)

    # 基础 BCE（和 Keras backend.binary_crossentropy 对应）
    if from_logits:
        bce = F.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction="none"
        )
    else:
        bce = F.binary_cross_entropy(
            y_pred, y_true, reduction="none"
        )

    # 默认倍数 1
    mult = torch.ones_like(y_true)

    # 对 region 内所有 logit 先整体乘 non_target_multiplier
    mult = mult + region * (non_target_multiplier - 1.0)

    # 再把正确 target 调回 target_multiplier（通常是 1）
    target_component = y_true * (target_multiplier - non_target_multiplier)
    if target_embedded_region:
        target_component = target_component * region
    mult = mult + target_component

    # 应用加权
    loss = bce * mult

    # 先在 word 维度上平均（对应 Keras 里的 axis）
    if axis < 0:
        axis = loss.dim() + axis
    loss = loss.mean(dim=axis)

    # 再对剩余维度整体平均，得到标量
    return loss.mean()


class DownWeightCompetitors(nn.Module):
    """
    PyTorch 版的 DownWeightCompetitors(by=c).

    和原 TF 版一样：
      - target 的损失不变
      - non-target 的损失除以 c
      - 只在 y_true 的 region mask=1 的时间段生效
    """

    def __init__(self, by: float, axis: int = -1, from_logits: bool = False):
        super().__init__()
        self.by = by
        self.axis = axis
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return cohort_weighted_loss(
            y_pred=y_pred,
            y_true=y_true,
            target_multiplier=1.0,
            non_target_multiplier=1.0 / self.by,  # = 1/c
            target_embedded_region=True,
            axis=self.axis,
            from_logits=self.from_logits,
        )

def framewise_bce_loss(logits_chunk,  # (B,Tc,V), raw logits
                       metas,         # list of dicts with "marks"
                       t0, t1,
                       device):
    B, Tc, V = logits_chunk.shape
    targets = torch.zeros(B, Tc, V, dtype=torch.float32, device=device)

    # 填充逐帧 one-hot 目标（静音默认全 0）
    for b in range(B):
        for (s, e, wid) in metas[b]["marks"]:
            if e <= t0 or s >= t1:
                continue
            s_loc = max(0, s - t0)
            e_loc = min(Tc, e - t0)
            if e_loc > s_loc:
                targets[b, s_loc:e_loc, wid] = 1.0

    # BCE with logits = sigmoid + BCE
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    

    loss = bce(logits_chunk, targets)
    return loss


def framewise_ce_loss(logits_chunk,  # (B,Tc,V) raw logits
                      metas,         # list of dicts with "marks"
                      t0, t1,
                      device,
                      blank_id: int = None):
    """
    If blank_id is not None, fill unlabelled frames with blank_id and train CE on all frames.
    Otherwise, compute CE only on the labelled frames (mask out others).
    """
    B, Tc, V = logits_chunk.shape

    if blank_id is not None:
        # Train on ALL frames (word frames = wid; silence/unlabelled = blank_id)
        targets = torch.full((B, Tc), blank_id, dtype=torch.long, device=device)
        for b in range(B):
            for (s, e, wid) in metas[b]["marks"]:
                s_loc = max(0, s - t0)
                e_loc = min(Tc, e - t0)
                if e_loc > s_loc:
                    targets[b, s_loc:e_loc] = int(wid)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits_chunk.transpose(1,2), targets)
        # (B,Tc,V) -> (B,V,Tc) for CE over last dim
        return loss
    else:
        # Train ONLY on labelled frames (word frames); ignore others
        all_idx = []
        all_tgt = []
        for b in range(B):
            for (s, e, wid) in metas[b]["marks"]:
                s_loc = max(0, s - t0)
                e_loc = min(Tc, e - t0)
                if e_loc > s_loc:
                    # collect (b, t) indices and their class
                    ts = torch.arange(s_loc, e_loc, device=device)
                    bs = torch.full((len(ts),), b, dtype=torch.long, device=device)
                    all_idx.append(torch.stack([bs, ts], dim=1))  # (L,2)
                    all_tgt.append(torch.full((len(ts),), int(wid), dtype=torch.long, device=device))
        if not all_idx:
            return logits_chunk.sum()*0.0  # no loss this chunk
        idx = torch.cat(all_idx, dim=0)         # (N, 2)
        tgt = torch.cat(all_tgt, dim=0)         # (N,)
        # gather logits at (b,t, :)
        gathered = logits_chunk[idx[:,0], idx[:,1], :]  # (N,V)
        loss = nn.CrossEntropyLoss(reduction="mean")(gathered, tgt)
        return loss

class CTCLossWrapper(nn.Module):
    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,   # (T, B, V)
        targets: torch.Tensor,     # (sumU,)
        input_lens: torch.Tensor,  # (B,)
        target_lens: torch.Tensor, # (B,)
    ) -> torch.Tensor:
        return self.ctc(log_probs, targets, input_lens, target_lens)


class ModifiedCTCLoss(nn.Module):
    """
    一个“可科研扩展”的示例：CTC + 长度归一项（避免长句子主导梯度）。
    你也可以在这里塞入你自己的 competitor down-weight / label smoothing 等想法。
    """
    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True, alpha: float = 0.5) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity, reduction="none")
        self.alpha = alpha

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lens: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        per_ex = self.ctc(log_probs, targets, input_lens, target_lens)  # (B,)
        norm = (target_lens.float().clamp_min(1.0) ** self.alpha)       # (B,)
        loss = (per_ex / norm).mean()
        return loss


def build_loss(loss_name: str, blank_idx: int) -> nn.Module:
    name = loss_name.lower()
    if name in ["ctc", "baseline"]:
        return CTCLossWrapper(blank_idx=blank_idx)
    if name in ["modified_loss", "modified_ctc"]:
        return ModifiedCTCLoss(blank_idx=blank_idx, alpha=0.5)
    raise ValueError(f"Unknown loss: {loss_name}")

