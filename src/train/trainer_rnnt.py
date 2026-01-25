# train/trainer_rnnt.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.rnnt import RNNTModel
from speech_dataset.vocab import build_librispeech_char_vocab, CharVocab
from speech_dataset.librispeech import LibriSpeechASR, collate_rnnt
from train.eva_rnnt import rnnt_greedy_decode, eval_metrics
from utils import AverageMeter, count_parameters, save_json, set_seed, setup_logger


def _build_rnnt_loss(blank_idx: int):
    from torchaudio.functional import rnnt_loss

    def loss_fn(logits, targets, logit_lens, target_lens):
        # logits: float32 更稳（AMP 下尤其重要）
        logits = logits.float()

        # rnnt_loss 要求 targets/int32
        targets_i32 = targets.to(torch.int32)
        logit_lens_i32 = logit_lens.to(torch.int32)
        target_lens_i32 = target_lens.to(torch.int32)

        return rnnt_loss(
            logits, targets_i32, logit_lens_i32, target_lens_i32,
            blank=blank_idx, reduction="mean"
        )

    return loss_fn


def _build_loaders(cfg: TrainConfig, vocab: CharVocab):
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

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_rnnt,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_rnnt,
    )
    return train_loader, valid_loader


@torch.no_grad()
def _run_eval(cfg: TrainConfig, model: RNNTModel, loader: DataLoader, rnnt_loss_fn, device: str, vocab: CharVocab, logger):
    model.eval()
    loss_meter = AverageMeter()
    all_wers, all_cers = [], []

    for batch in loader:
        feats = batch["feats"].to(device)
        feat_lens = batch["feat_lens"].to(device)
        targets = batch["targets"].to(device)
        target_lens = batch["target_lens"].to(device)
        texts = batch["texts"]

        logits, out_lens, _ = model(feats, feat_lens, targets, target_lens)
        loss = rnnt_loss_fn(logits, targets, out_lens, target_lens)
        loss_meter.update(float(loss.item()), n=feats.size(0))

        hyps = rnnt_greedy_decode(model, feats, feat_lens, vocab)
        w, c = eval_metrics(texts, hyps)
        all_wers.append(w)
        all_cers.append(c)

    return loss_meter.avg, float(sum(all_wers) / max(1, len(all_wers))), float(sum(all_cers) / max(1, len(all_cers)))


def run_once(cfg: TrainConfig, device: str) -> None:
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)

    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")

    vocab = build_librispeech_char_vocab()
    logger.info(f"Vocab size: {vocab.size} (blank={vocab.blank_idx})")

    train_loader, valid_loader = _build_loaders(cfg, vocab)

    # TrainConfig; below uses safe defaults if missing
    enc_sub = getattr(cfg, "rnnt_enc_subsample", 4)
    pred_embed = getattr(cfg, "rnnt_pred_embed", 256)
    pred_hidden = getattr(cfg, "rnnt_pred_hidden", 512)
    joint_dim = getattr(cfg, "rnnt_joint_dim", 512)

    model = RNNTModel(
        in_dim=cfg.n_mels,
        vocab_size=vocab.size,
        blank_idx=vocab.blank_idx,
        enc_hidden=cfg.lstm_hidden,
        enc_layers=cfg.lstm_layers,
        enc_bidir=cfg.bidirectional,
        enc_dropout=cfg.dropout,
        enc_subsample=enc_sub,
        pred_embed=pred_embed,
        pred_hidden=pred_hidden,
        pred_layers=2,
        pred_dropout=cfg.dropout,
        joint_dim=joint_dim,
    ).to(device)

    logger.info(f"Model params: {count_parameters(model):,}")

    rnnt_loss_fn = _build_rnnt_loss(vocab.blank_idx)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_wer = float("inf")
    logger.info("Start training...")

    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()

        for it, batch in enumerate(train_loader, start=1):
            feats = batch["feats"].to(device)
            feat_lens = batch["feat_lens"].to(device)
            targets = batch["targets"].to(device)
            target_lens = batch["target_lens"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                log_probs, out_lens, _ = model(feats, feat_lens, targets, target_lens)
                loss = rnnt_loss_fn(log_probs, targets, out_lens, target_lens)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))
            if it % cfg.log_every == 0:
                logger.info(f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f}")

        if (ep % cfg.eval_every_epochs) == 0:
            val_loss, val_wer, val_cer = _run_eval(cfg, model, valid_loader, rnnt_loss_fn, device, vocab, logger)
            logger.info(f"[VALID] Epoch {ep:03d} | loss {val_loss:.4f} | WER {val_wer:.4f} | CER {val_cer:.4f}")

            if val_wer < best_wer:
                best_wer = val_wer
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": ep,
                    "best_wer": best_wer,
                }
                torch.save(ckpt, cfg.ckpt_dir / "best.pt")
                logger.info(f"Saved best checkpoint (WER={best_wer:.4f})")

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep, "best_wer": best_wer},
            cfg.ckpt_dir / "last.pt",
        )

    save_json(cfg.run_dir / "result.json", {"best_wer": best_wer})
    logger.info("Done.")
