# train/trainer_librispeech.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.lstm_ctc import LSTMCTC
from speech_dataset.vocab import build_librispeech_char_vocab, CharVocab
from speech_dataset.librispeech import LibriSpeechASR, collate_librispeech
from network.loss import build_loss
from train.eva import evaluate_batch, EvalResult
from utils.utils import (
    AverageMeter,
    count_parameters,
    save_json,
    set_seed,
    setup_logger,
    to_device,
)


def _build_loaders(cfg: TrainConfig, vocab: CharVocab) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LibriSpeechASR(
        root=cfg.librispeech_root,
        subset=cfg.train_subset,
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_length_ms=cfg.win_length_ms,
        hop_length_ms=cfg.hop_length_ms,
        download=True,
        limit_samples=cfg.limit_train_samples,
    )
    valid_ds = LibriSpeechASR(
        root=cfg.librispeech_root,
        subset=cfg.valid_subset,
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_length_ms=cfg.win_length_ms,
        hop_length_ms=cfg.hop_length_ms,
        download=True,
        limit_samples=cfg.limit_valid_samples,
    )
    test_ds = LibriSpeechASR(
        root=cfg.librispeech_root,
        subset=cfg.test_subset,
        vocab=vocab,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        win_length_ms=cfg.win_length_ms,
        hop_length_ms=cfg.hop_length_ms,
        download=True,
        limit_samples=cfg.limit_test_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_librispeech,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_librispeech,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_librispeech,
    )
    return train_loader, valid_loader, test_loader


def _save_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_wer: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_wer": best_wer,
        },
        path,
    )


@torch.no_grad()
def _run_eval(cfg: TrainConfig, model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str, vocab: CharVocab, logger) -> EvalResult:
    model.eval()
    loss_meter = AverageMeter()

    all_wers = []
    all_cers = []

    for batch in loader:
        feats = batch.feats.to(device)
        feat_lens = batch.feat_lens.to(device)
        targets = batch.targets.to(device)
        target_lens = batch.target_lens.to(device)

        log_probs, out_lens = model(feats, feat_lens)
        loss = loss_fn(log_probs, targets, out_lens, target_lens)
        loss_meter.update(float(loss.item()), n=feats.size(0))

        _, wers, cers = evaluate_batch(log_probs, out_lens, batch.texts, vocab)
        all_wers.extend(wers)
        all_cers.extend(cers)

    mean_wer = float(sum(all_wers) / max(1, len(all_wers)))
    mean_cer = float(sum(all_cers) / max(1, len(all_cers)))

    return EvalResult(mean_loss=loss_meter.avg, wer=mean_wer, cer=mean_cer)


def run_once(cfg: TrainConfig, device: str) -> None:
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)

    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")

    vocab = build_librispeech_char_vocab()
    logger.info(f"Vocab size: {vocab.size} (blank={vocab.blank_idx})")

    train_loader, valid_loader, test_loader = _build_loaders(cfg, vocab)

    model = LSTMCTC(
        in_dim=cfg.n_mels,
        vocab_size=vocab.size,
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    loss_fn = build_loss(cfg.policy_name, blank_idx=vocab.blank_idx).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_wer = float("inf")

    logger.info("Start training...")
    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()

        for it, batch in enumerate(train_loader, start=1):
            feats = batch.feats.to(device)
            feat_lens = batch.feat_lens.to(device)
            targets = batch.targets.to(device)
            target_lens = batch.target_lens.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                log_probs, out_lens = model(feats, feat_lens)
                loss = loss_fn(log_probs, targets, out_lens, target_lens)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))

            if it % cfg.log_every == 0:
                logger.info(f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f}")

        # eval
        if (ep % cfg.eval_every_epochs) == 0:
            val_res = _run_eval(cfg, model, valid_loader, loss_fn, device, vocab, logger)
            logger.info(
                f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} | WER {val_res.wer:.4f} | CER {val_res.cer:.4f}"
            )

            # save best
            if val_res.wer < best_wer:
                best_wer = val_res.wer
                _save_ckpt(cfg.ckpt_dir / "best.pt", model, optimizer, ep, best_wer)
                logger.info(f"Saved best checkpoint (WER={best_wer:.4f})")

        # always save last
        _save_ckpt(cfg.ckpt_dir / "last.pt", model, optimizer, ep, best_wer)

    # final test with best
    logger.info("Testing best checkpoint...")
    ckpt_path = cfg.ckpt_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    test_res = _run_eval(cfg, model, test_loader, loss_fn, device, vocab, logger)
    logger.info(
        f"[TEST] loss {test_res.mean_loss:.4f} | WER {test_res.wer:.4f} | CER {test_res.cer:.4f}"
    )

    save_json(cfg.run_dir / "result.json", {"best_wer": best_wer, "test": asdict(test_res)})
    logger.info("Done.")
