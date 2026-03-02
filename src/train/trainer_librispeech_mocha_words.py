# train/trainer_librispeech_mocha_words.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.las_monotonic import CausalLASMonotonic

from utils.utils import (
    AverageMeter,
    count_parameters,
    save_json,
    set_seed,
    setup_logger,
)

from speech_dataset.librispeech_aligned_words import (
    build_word_vocab_from_manifest,
    LibriSpeechAlignedWords,
)
from speech_dataset.collate_aligned_words import collate_aligned_words

IGNORE_INDEX = -100


@dataclass
class EvalResultSeq:
    mean_loss: float
    tok_acc: float
    n_tok: int


def _targets_to_word_seq(t: torch.Tensor) -> List[int]:
    """
    t: (T,) with word_id on tail frames, IGNORE elsewhere.
    Return compressed word sequence in time order (remove consecutive duplicates).
    """
    y = t[t != IGNORE_INDEX]
    if y.numel() == 0:
        return []
    y_list = y.tolist()
    seq = [int(y_list[0])]
    for a in y_list[1:]:
        a = int(a)
        if a != seq[-1]:
            seq.append(a)
    return seq


def _make_teacher_forcing_batch(
    targets_frame: torch.Tensor,  # (B,T)
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_words: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert frame targets -> word sequence targets -> y_in/y_out for teacher forcing.

    Returns:
      y_in:  (B,U)  [BOS, w1, w2, ..., PAD]
      y_out: (B,U)  [w1,  w2, ..., EOS, PAD]
      y_lens:(B,)   number of valid steps in y_out (incl EOS)
    """
    B, T = targets_frame.shape
    seqs: List[List[int]] = []
    for b in range(B):
        seq = _targets_to_word_seq(targets_frame[b])
        if len(seq) > max_words:
            seq = seq[:max_words]
        seqs.append(seq)

    # U = max(seq_len)+1 for EOS (and BOS in y_in)
    max_len = max((len(s) for s in seqs), default=0)
    U = max_len + 1  # +1 for EOS in y_out

    y_in = torch.full((B, U), pad_id, dtype=torch.long, device=targets_frame.device)
    y_out = torch.full((B, U), pad_id, dtype=torch.long, device=targets_frame.device)
    y_lens = torch.zeros((B,), dtype=torch.long, device=targets_frame.device)

    for b, seq in enumerate(seqs):
        L = len(seq)
        # y_in: BOS + seq
        y_in[b, 0] = bos_id
        if L > 0:
            y_in[b, 1 : 1 + L] = torch.tensor(seq, dtype=torch.long, device=targets_frame.device)

        # y_out: seq + EOS
        if L > 0:
            y_out[b, 0:L] = torch.tensor(seq, dtype=torch.long, device=targets_frame.device)
        y_out[b, L] = eos_id

        y_lens[b] = L + 1  # include EOS

    return y_in, y_out, y_lens


def _build_loaders(cfg: TrainConfig, vocab, tail_frames: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    if cfg.librispeech_manifest is None:
        raise ValueError("TrainConfig.librispeech_manifest must be set.")

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
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val": best_val},
        path,
    )


@torch.no_grad()
def _run_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_words: int,
) -> EvalResultSeq:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for batch in loader:
        feats = batch.feats.to(device)          # (B,T,F)
        feat_lens = batch.feat_lens.to(device)  # (B,)
        targets = batch.targets.to(device)      # (B,T), IGNORE elsewhere

        y_in, y_out, _ = _make_teacher_forcing_batch(
            targets_frame=targets,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_words=max_words,
        )

        logits = model(feats, feat_lens, y_in)  # (B,U,V)
        B, U, V = logits.shape
        loss = loss_fn(logits.reshape(-1, V), y_out.reshape(-1))
        loss_meter.update(float(loss.item()), n=B)

        pred = logits.argmax(dim=-1)  # (B,U)
        mask = y_out != pad_id
        if mask.any():
            correct += int((pred[mask] == y_out[mask]).sum().item())
            total += int(mask.sum().item())

    acc = float(correct / total) if total > 0 else 0.0
    return EvalResultSeq(mean_loss=loss_meter.avg, tok_acc=acc, n_tok=total)


def run_once(cfg: TrainConfig, device: str) -> None:
    """
    Strictly-causal semantic baseline:
      - encoder: UniLSTM on gammatone (64ch), optional causal subsample by stacking
      - attention: expected monotonic attention (no global lookahead)
      - decoder: LSTM language model predicting word sequence (derived from word-tail frame labels)
    """
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)
    set_seed(cfg.seed)

    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")
    logger.info(f"Task: {cfg.task_name}")

    env_sr = getattr(cfg, "env_sr", 100)
    tail_ms = getattr(cfg, "tail_ms", 100)
    topk = getattr(cfg, "word_vocab_topk", 20000)
    feat_dim = getattr(cfg, "coch_feat_dim", 64)
    tail_frames = max(1, int(round((tail_ms / 1000.0) * env_sr)))

    manifest_path = Path(cfg.librispeech_manifest)
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Word supervision tail: {tail_ms} ms => {tail_frames} frames @ env_sr={env_sr}")
    logger.info(f"batch_size={cfg.batch_size} (if OOM/slow, drop to 4~8)")

    vocab = build_word_vocab_from_manifest(manifest_path, topk=topk)
    logger.info(f"Word vocab size: {vocab.size} (unk_id={vocab.unk_id}) topk={topk}")

    # add special tokens for seq2seq
    pad_id = vocab.size
    bos_id = vocab.size + 1
    eos_id = vocab.size + 2
    seq_vocab_size = vocab.size + 3

    train_loader, valid_loader, test_loader = _build_loaders(cfg, vocab, tail_frames)

    model = CausalLASMonotonic(
        in_dim=feat_dim,
        vocab_size=seq_vocab_size,
        enc_hidden=cfg.lstm_hidden,
        enc_layers=cfg.lstm_layers,
        dec_embed=cfg.las_dec_embed,
        dec_hidden=cfg.las_dec_hidden,
        dec_layers=cfg.las_dec_layers,
        attn_dim=cfg.las_attn_dim,
        dropout=cfg.dropout,
        enc_subsample=cfg.las_enc_subsample,
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")
    logger.info("Start training (Causal LAS + Monotonic Attention)...")

    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for it, batch in enumerate(train_loader, start=1):
            feats = batch.feats.to(device)
            feat_lens = batch.feat_lens.to(device)
            targets = batch.targets.to(device)

            y_in, y_out, _ = _make_teacher_forcing_batch(
                targets_frame=targets,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                max_words=cfg.las_max_words,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                logits = model(feats, feat_lens, y_in)   # (B,U,V)
                B, U, V = logits.shape
                loss = loss_fn(logits.reshape(-1, V), y_out.reshape(-1))

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = y_out != pad_id
                if mask.any():
                    correct += int((pred[mask] == y_out[mask]).sum().item())
                    total += int(mask.sum().item())

            if it % cfg.log_every == 0:
                acc = float(correct / total) if total > 0 else 0.0
                logger.info(
                    f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f} "
                    f"| tok_acc {acc:.4f} | n_tok {total}"
                )

        # ---- eval ----
        if (ep % cfg.eval_every_epochs) == 0:
            val_res = _run_eval(
                model=model,
                loader=valid_loader,
                loss_fn=loss_fn,
                device=device,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                max_words=cfg.las_max_words,
            )
            logger.info(f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} | tok_acc {val_res.tok_acc:.4f} | n_tok {val_res.n_tok}")

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

    test_res = _run_eval(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_words=cfg.las_max_words,
    )
    logger.info(f"[TEST] loss {test_res.mean_loss:.4f} | tok_acc {test_res.tok_acc:.4f} | n_tok {test_res.n_tok}")

    save_json(cfg.run_dir / "result.json", {"best_val_loss": best_val, "test": asdict(test_res)})
    logger.info("Done.")
