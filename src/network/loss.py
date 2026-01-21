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