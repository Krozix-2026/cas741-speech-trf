# train/trainer_librispeech_phonemes_srv.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.lstm_frame_srv import LSTMFrameSRV

from utils.utils import (
    AverageMeter,
    count_parameters,
    save_json,
    set_seed,
    setup_logger,
)

from utils.srv import make_srv_table

from speech_dataset.librispeech_aligned_phonemes import (
    build_phone_vocab_from_manifest,
    LibriSpeechAlignedPhonemes,
)
from speech_dataset.collate_aligned_phonemes import collate_aligned_phonemes

IGNORE_INDEX = -100


@dataclass
class EvalResultSRV:
    mean_loss: float
    mean_cos: float
    sup_frames: int
    nn_acc: float


def _save_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val": best_val},
        path,
    )


def _srv_loss_and_cos(
    y: torch.Tensor,            # (B,T,D)
    targets: torch.Tensor,      # (B,T) phone ids or IGNORE
    srv_table: torch.Tensor,    # (V,D) normalized
    loss_type: str,
) -> tuple[torch.Tensor, float, int]:
    mask = targets != IGNORE_INDEX
    if not mask.any():
        return y.sum() * 0.0, 0.0, 0

    y_lab = y[mask]                      # (N,D)
    pid = targets[mask].long()           # (N,)
    t_lab = srv_table[pid]               # (N,D)

    y_n = F.normalize(y_lab.float(), dim=-1, eps=1e-8)
    t_n = t_lab.float()
    cos = (y_n * t_n).sum(dim=-1)
    mean_cos = float(cos.mean().item())

    if loss_type.lower() == "cosine":
        loss = (1.0 - cos).mean()
    elif loss_type.lower() == "mse":
        loss = F.mse_loss(y_n, t_n)
    else:
        raise ValueError(f"Unknown srv_loss: {loss_type} (cosine|mse)")

    return loss, mean_cos, int(pid.numel())


@torch.no_grad()
def _nn_acc_optional(y, targets, srv_table, max_frames: int) -> float:
    mask = targets != IGNORE_INDEX
    if not mask.any():
        return 0.0
    y_lab = y[mask].float()
    pid = targets[mask].long()
    if y_lab.shape[0] > max_frames:
        y_lab = y_lab[:max_frames]
        pid = pid[:max_frames]
    y_n = F.normalize(y_lab, dim=-1, eps=1e-8)
    sims = y_n @ srv_table.t().float()
    pred = sims.argmax(dim=-1)
    return float((pred == pid).float().mean().item())


@torch.no_grad()
def _run_eval_srv(model, loader, device, srv_table, cfg) -> EvalResultSRV:
    model.eval()
    loss_meter = AverageMeter()
    cos_meter = AverageMeter()
    total = 0

    for batch in loader:
        feats = batch.feats.to(device)
        feat_lens = batch.feat_lens.to(device)
        targets = batch.targets.to(device)

        y, _ = model(feats, feat_lens)
        loss, mean_cos, n = _srv_loss_and_cos(y, targets, srv_table, cfg.srv_loss)
        loss_meter.update(float(loss.item()), n=feats.size(0))
        cos_meter.update(mean_cos, n=max(1, n))
        total += n

    nn_acc = 0.0
    if getattr(cfg, "srv_eval_nn", False):
        for batch in loader:
            feats = batch.feats.to(device)
            feat_lens = batch.feat_lens.to(device)
            targets = batch.targets.to(device)
            y, _ = model(feats, feat_lens)
            nn_acc = _nn_acc_optional(y, targets, srv_table, cfg.srv_eval_max_frames)
            break

    return EvalResultSRV(mean_loss=loss_meter.avg, mean_cos=cos_meter.avg, sup_frames=total, nn_acc=nn_acc)


def _build_loaders(cfg: TrainConfig, vocab, tail_frames: int, center_frames: int):
    if cfg.librispeech_manifest is None:
        raise ValueError("TrainConfig.librispeech_manifest must be set (phoneme-augmented manifest).")
    manifest_path = Path(cfg.librispeech_manifest)

    label_region = getattr(cfg, "phone_label_region", "all")

    train_ds = LibriSpeechAlignedPhonemes(
        manifest_path=manifest_path,
        subset=cfg.train_subset,
        vocab=vocab,
        label_region=label_region,
        tail_frames=tail_frames,
        center_frames=center_frames,
        limit_samples=cfg.limit_train_samples,
    )
    valid_ds = LibriSpeechAlignedPhonemes(
        manifest_path=manifest_path,
        subset=cfg.valid_subset,
        vocab=vocab,
        label_region=label_region,
        tail_frames=tail_frames,
        center_frames=center_frames,
        limit_samples=cfg.limit_valid_samples,
    )
    test_ds = LibriSpeechAlignedPhonemes(
        manifest_path=manifest_path,
        subset=cfg.test_subset,
        vocab=vocab,
        label_region=label_region,
        tail_frames=tail_frames,
        center_frames=center_frames,
        limit_samples=cfg.limit_test_samples,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_aligned_phonemes,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_aligned_phonemes,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_aligned_phonemes,
    )
    return train_loader, valid_loader, test_loader


def run_once(cfg: TrainConfig, device: str) -> None:
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)

    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")

    env_sr = getattr(cfg, "env_sr", 100)
    feat_dim = getattr(cfg, "coch_feat_dim", 64)

    # phoneme label region params
    phone_label_region = getattr(cfg, "phone_label_region", "all")
    phone_tail_ms = int(getattr(cfg, "phone_tail_ms", 30))
    phone_center_ms = int(getattr(cfg, "phone_center_ms", 30))
    tail_frames = max(1, int(round((phone_tail_ms / 1000.0) * env_sr)))
    center_frames = max(1, int(round((phone_center_ms / 1000.0) * env_sr)))

    manifest_path = Path(cfg.librispeech_manifest)
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Phone supervision region: {phone_label_region} (tail={tail_frames}f, center={center_frames}f) env_sr={env_sr}")

    vocab = build_phone_vocab_from_manifest(manifest_path, topk=None)
    logger.info(f"Phone vocab size: {vocab.size} (unk_id={vocab.unk_id})")

    train_loader, valid_loader, test_loader = _build_loaders(cfg, vocab, tail_frames, center_frames)

    # SRV table
    D = int(cfg.srv_dim)
    k = int(cfg.srv_k)
    logger.info(f"SRV: dim={D}, k={k}, value={cfg.srv_value}, seed={cfg.srv_seed}, loss={cfg.srv_loss}")

    srv_table_cpu = make_srv_table(
        vocab_size=vocab.size,
        dim=D,
        k=k,
        seed=int(cfg.srv_seed),
        value=str(cfg.srv_value),
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )
    srv_table = srv_table_cpu.to(device, non_blocking=True)
    torch.save(
        {"srv_dim": D, "srv_k": k, "srv_value": cfg.srv_value, "srv_seed": cfg.srv_seed, "table": srv_table_cpu.cpu()},
        cfg.run_dir / "srv_table.pt",
    )

    # model
    model = LSTMFrameSRV(
        in_dim=feat_dim,
        out_dim=D,
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        bidirectional=False,
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")
    logger.info("Start training (phoneme SRV regression)...")

    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()
        cos_meter = AverageMeter()
        total = 0

        for it, batch in enumerate(train_loader, start=1):
            feats = batch.feats.to(device, non_blocking=True)
            feat_lens = batch.feat_lens.to(device, non_blocking=True)
            targets = batch.targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                y, _ = model(feats, feat_lens)
                loss, mean_cos, n = _srv_loss_and_cos(y, targets, srv_table, cfg.srv_loss)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))
            cos_meter.update(mean_cos, n=max(1, n))
            total += n

            if it % cfg.log_every == 0:
                logger.info(
                    f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f} | mean_cos {cos_meter.avg:.4f} | sup_frames {total}"
                )

        if (ep % cfg.eval_every_epochs) == 0:
            val_res = _run_eval_srv(model, valid_loader, device, srv_table, cfg)
            logger.info(
                f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} | mean_cos {val_res.mean_cos:.4f} "
                f"| sup_frames {val_res.sup_frames} | nn_acc {val_res.nn_acc:.4f}"
            )
            if val_res.mean_loss < best_val:
                best_val = val_res.mean_loss
                _save_ckpt(cfg.ckpt_dir / "best.pt", model, optimizer, ep, best_val)
                logger.info(f"Saved best checkpoint (val_loss={best_val:.4f})")

        _save_ckpt(cfg.ckpt_dir / "last.pt", model, optimizer, ep, best_val)

    logger.info("Testing best checkpoint...")
    ckpt_path = cfg.ckpt_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    test_res = _run_eval_srv(model, test_loader, device, srv_table, cfg)
    logger.info(
        f"[TEST] loss {test_res.mean_loss:.4f} | mean_cos {test_res.mean_cos:.4f} | sup_frames {test_res.sup_frames} | nn_acc {test_res.nn_acc:.4f}"
    )
    save_json(cfg.run_dir / "result.json", {"best_val_loss": best_val, "test": asdict(test_res)})
    logger.info("Done.")