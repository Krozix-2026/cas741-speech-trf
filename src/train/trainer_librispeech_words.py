# train/trainer_librispeech_words.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.lstm_frame_classifier import LSTMFrameClassifier

from utils.utils import (
    AverageMeter,
    count_parameters,
    save_json,
    set_seed,
    setup_logger,
)

# You need to provide these modules (simple, small):
# - build_word_vocab_from_manifest(manifest_path, topk) -> vocab with .stoi/.itos/.unk_id/.size
# - LibriSpeechAlignedWords(manifest_path, subset, vocab, tail_frames, limit_samples)
# - collate_aligned_words(items) -> batch with .feats/.feat_lens/.targets/.utt_ids
from speech_dataset.librispeech_aligned_words import (
    build_word_vocab_from_manifest,
    LibriSpeechAlignedWords,
)
from speech_dataset.collate_aligned_words import collate_aligned_words

IGNORE_INDEX = -100


@dataclass
class EvalResultWords:
    mean_loss: float
    sup_frame_acc: float
    sup_frames: int


def _build_loaders_words(cfg: TrainConfig, vocab, tail_frames: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build loaders from manifest (NOT torchaudio mel features).
    """
    if cfg.librispeech_manifest is None:
        raise ValueError("TrainConfig.librispeech_manifest must be set for word-aligned training.")

    manifest_path = Path(cfg.librispeech_manifest)

    train_ds = LibriSpeechAlignedWords(
        manifest_path=manifest_path,
        subset=cfg.train_subset,
        vocab=vocab,
        tail_frames=tail_frames,
        limit_samples=cfg.limit_train_samples,
    )
    valid_ds = LibriSpeechAlignedWords(
        manifest_path=manifest_path,
        subset=cfg.valid_subset,
        vocab=vocab,
        tail_frames=tail_frames,
        limit_samples=cfg.limit_valid_samples,
    )
    test_ds = LibriSpeechAlignedWords(
        manifest_path=manifest_path,
        subset=cfg.test_subset,
        vocab=vocab,
        tail_frames=tail_frames,
        limit_samples=cfg.limit_test_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_aligned_words,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_aligned_words,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_aligned_words,
    )
    return train_loader, valid_loader, test_loader


def _save_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


@torch.no_grad()
def _run_eval_words(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    logger,
) -> EvalResultWords:
    model.eval()
    loss_meter = AverageMeter()

    correct = 0
    total = 0

    for batch in loader:
        feats = batch.feats.to(device)         # (B, Tmax, F)
        feat_lens = batch.feat_lens.to(device) # (B,)
        targets = batch.targets.to(device)     # (B, Tmax), -100 where ignored

        logits, out_lens = model(feats, feat_lens)  # (B, Tmax, V)

        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(-1, V), targets.reshape(-1))
        loss_meter.update(float(loss.item()), n=B)

        # supervised-frame accuracy (only where targets != IGNORE)
        mask = targets != IGNORE_INDEX
        if mask.any():
            pred = logits.argmax(dim=-1)
            correct += int((pred[mask] == targets[mask]).sum().item())
            total += int(mask.sum().item())

    acc = float(correct / total) if total > 0 else 0.0
    return EvalResultWords(mean_loss=loss_meter.avg, sup_frame_acc=acc, sup_frames=total)


def run_once(cfg: TrainConfig, device: str) -> None:
    """
    Word-aligned training on cochleagrams:
      input: coch (T, 64)
      target: word_id on word-tail frames; ignore elsewhere
    """
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)

    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")

    # ---- hyperparams specific to word-aligned training ----
    # Require these fields in cfg (add them to TrainConfig):
    # - librispeech_manifest: Path
    # - env_sr: int = 100
    # - tail_ms: int = 100
    # - word_vocab_topk: int = 20000
    # - coch_feat_dim: int = 64
    env_sr = getattr(cfg, "env_sr", 100)
    tail_ms = getattr(cfg, "tail_ms", 100)
    topk = getattr(cfg, "word_vocab_topk", 20000)
    feat_dim = getattr(cfg, "coch_feat_dim", 64)

    tail_frames = int(round((tail_ms / 1000.0) * env_sr))
    tail_frames = max(1, tail_frames)

    manifest_path = Path(cfg.librispeech_manifest)
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Word supervision tail: {tail_ms} ms => {tail_frames} frames at env_sr={env_sr}")

    # ---- vocab ----
    vocab = build_word_vocab_from_manifest(manifest_path, topk=topk)
    logger.info(f"Word vocab size: {vocab.size} (unk_id={vocab.unk_id}) topk={topk}")

    # ---- loaders ----
    train_loader, valid_loader, test_loader = _build_loaders_words(cfg, vocab, tail_frames)

    # ---- model (UNIDIRECTIONAL) ----
    model = LSTMFrameClassifier(
        in_dim=feat_dim,
        vocab_size=vocab.size,
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        bidirectional=False,  # enforce
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    # ---- loss ----
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX).to(device)

    # ---- optim ----
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")  # track best by val loss (or you could use 1-acc)

    logger.info("Start training (word-aligned frame classification)...")
    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for it, batch in enumerate(train_loader, start=1):
            feats = batch.feats.to(device)
            # print("feats:", feats.shape)#32, 1658, 64]
            feat_lens = batch.feat_lens.to(device)
            # print("feat_lens:", feat_lens.shape)#32
            targets = batch.targets.to(device)
            print("targets:", targets.shape)#[32, 1658]

            t = batch.targets.to(device)
            # print("unique:", torch.unique(t)[:20], " ... n_unique=", torch.unique(t).numel())
            # print("labeled_frames:", (t != -100).sum().item())
            # print("any_labeled:", bool((t != -100).any().item()))
            
            
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                logits, out_lens = model(feats, feat_lens)
                # print("logits:", logits)#[32, 1658, 20001]
                # print("out_lens:", out_lens)#[32]
                B, T, V = logits.shape
                # print("logits.reshape(-1, V):", logits.reshape(-1, V).shape)#[32x1658, 20001]
                loss = loss_fn(logits.reshape(-1, V), targets.reshape(-1))

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))

            # train supervised-frame accuracy (quick sanity)
            with torch.no_grad():
                mask = targets != IGNORE_INDEX
                if mask.any():
                    pred = logits.argmax(dim=-1)
                    correct += int((pred[mask] == targets[mask]).sum().item())
                    total += int(mask.sum().item())

            if it % cfg.log_every == 0:
                acc = float(correct / total) if total > 0 else 0.0
                logger.info(f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f} | sup_acc {acc:.4f} | sup_frames {total}")

        # ---- eval ----
        if (ep % cfg.eval_every_epochs) == 0:
            val_res = _run_eval_words(model, valid_loader, loss_fn, device, logger)
            logger.info(
                f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} | sup_acc {val_res.sup_frame_acc:.4f} | sup_frames {val_res.sup_frames}"
            )

            if val_res.mean_loss < best_val:
                best_val = val_res.mean_loss
                _save_ckpt(cfg.ckpt_dir / "best.pt", model, optimizer, ep, best_val)
                logger.info(f"Saved best checkpoint (val_loss={best_val:.4f})")

        _save_ckpt(cfg.ckpt_dir / "last.pt", model, optimizer, ep, best_val)

    # ---- test best ----
    logger.info("Testing best checkpoint...")
    ckpt_path = cfg.ckpt_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    test_res = _run_eval_words(model, test_loader, loss_fn, device, logger)
    logger.info(
        f"[TEST] loss {test_res.mean_loss:.4f} | sup_acc {test_res.sup_frame_acc:.4f} | sup_frames {test_res.sup_frames}"
    )

    save_json(cfg.run_dir / "result.json", {"best_val_loss": best_val, "test": asdict(test_res)})
    logger.info("Done.")