# train/trainer_librispeech_las_words.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.las_global import LASGlobal

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
    wer: float
    n_tok: int
    n_words: int



def _strip_special(seq: torch.Tensor, pad_id: int, eos_id: int, bos_id: int | None = None) -> list[int]:
    """Convert 1D tensor -> list[int], stop at EOS, drop PAD/BOS."""
    out: list[int] = []
    for x in seq.tolist():
        if x == pad_id:
            continue
        if bos_id is not None and x == bos_id:
            continue
        if x == eos_id:
            break
        out.append(int(x))
    return out


def _edit_distance(ref: list[int], hyp: list[int]) -> int:
    """Levenshtein distance."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,        # deletion
                dp[j - 1] + 1,    # insertion
                prev + cost       # substitution
            )
            prev = cur
    return dp[m]


def _batch_wer(pred_ids: torch.Tensor, tgt_ids: torch.Tensor, pad_id: int, eos_id: int, bos_id: int) -> tuple[int, int]:
    """
    pred_ids/tgt_ids: (B,U)
    returns: (total_edits, total_ref_words)
    """
    B = pred_ids.size(0)
    edits = 0
    nref = 0
    for b in range(B):
        hyp = _strip_special(pred_ids[b], pad_id=pad_id, eos_id=eos_id, bos_id=bos_id)
        ref = _strip_special(tgt_ids[b], pad_id=pad_id, eos_id=eos_id, bos_id=None)  # y_out通常不含BOS
        edits += _edit_distance(ref, hyp)
        nref += len(ref)
    return edits, nref



def _targets_to_word_seq(t: torch.Tensor) -> List[int]:
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
    B, T = targets_frame.shape
    seqs: List[List[int]] = []
    for b in range(B):
        seq = _targets_to_word_seq(targets_frame[b])
        if len(seq) > max_words:
            seq = seq[:max_words]
        seqs.append(seq)

    max_len = max((len(s) for s in seqs), default=0)
    U = max_len + 1  # +1 for EOS in y_out

    y_in = torch.full((B, U), pad_id, dtype=torch.long, device=targets_frame.device)
    y_out = torch.full((B, U), pad_id, dtype=torch.long, device=targets_frame.device)
    y_lens = torch.zeros((B,), dtype=torch.long, device=targets_frame.device)

    for b, seq in enumerate(seqs):
        L = len(seq)
        y_in[b, 0] = bos_id
        if L > 0:
            y_in[b, 1 : 1 + L] = torch.tensor(seq, dtype=torch.long, device=targets_frame.device)

        if L > 0:
            y_out[b, 0:L] = torch.tensor(seq, dtype=torch.long, device=targets_frame.device)
        y_out[b, L] = eos_id
        y_lens[b] = L + 1

    return y_in, y_out, y_lens


def _build_loaders(cfg: TrainConfig, vocab, tail_frames: int):
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
    word_edits = 0
    word_ref = 0
    
    
    
        
    

    for batch in loader:
        feats = batch.feats.to(device)
        feat_lens = batch.feat_lens.to(device)
        targets = batch.targets.to(device)

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
        
        
        # with torch.no_grad():
        #     pred = logits.argmax(dim=-1)  # (B,U)
        #     # eos rate
        #     eos_hit = (pred == eos_id).any(dim=1).float().mean().item()
        #     # lengths
        #     # ref_len: count non-pad until eos in y_out
        #     ref_lens = []
        #     hyp_lens = []
        #     for b in range(pred.size(0)):
        #         ref = _strip_special(y_out[b], pad_id, eos_id, bos_id=None)
        #         hyp = _strip_special(pred[b], pad_id, eos_id, bos_id=bos_id)
        #         ref_lens.append(len(ref))
        #         hyp_lens.append(len(hyp))
        #     print(f"[dbg] eos_hit={eos_hit:.3f}  ref_len={sum(ref_lens)/len(ref_lens):.1f}  hyp_len={sum(hyp_lens)/len(hyp_lens):.1f}")

        
        

        pred = logits.argmax(dim=-1)
        
        be, br = _batch_wer(pred, y_out, pad_id=pad_id, eos_id=eos_id, bos_id=bos_id)
        word_edits += be
        word_ref += br
        
        mask = y_out != pad_id
        if mask.any():
            correct += int((pred[mask] == y_out[mask]).sum().item())
            total += int(mask.sum().item())

    wer = float(word_edits / word_ref) if word_ref > 0 else 0.0
    acc = float(correct / total) if total > 0 else 0.0
    return EvalResultSeq(mean_loss=loss_meter.avg, tok_acc=acc, wer=wer, n_tok=total, n_words=word_ref)


def run_once(cfg: TrainConfig, device: str) -> None:
    """
    Classic LAS (global attention) baseline for word-aligned LibriSpeech coch features:
      - encoder: (Bi)LSTM on coch features (+ stacking subsample)
      - attention: global additive attention
      - decoder: LSTM LM predicting word sequence (derived from word-tail frame labels)
    """
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)
    set_seed(cfg.seed)

    env_sr = getattr(cfg, "env_sr", 100)
    tail_ms = getattr(cfg, "tail_ms", 100)
    topk = getattr(cfg, "word_vocab_topk", 20000)
    feat_dim = getattr(cfg, "coch_feat_dim", 64)
    tail_frames = max(1, int(round((tail_ms / 1000.0) * env_sr)))

    manifest_path = Path(cfg.librispeech_manifest)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Word supervision tail: {tail_ms} ms => {tail_frames} frames @ env_sr={env_sr}")
    logger.info(f"batch_size={cfg.batch_size}")

    vocab = build_word_vocab_from_manifest(manifest_path, topk=topk)
    logger.info(f"Word vocab size: {vocab.size} (unk_id={vocab.unk_id}) topk={topk}")

    pad_id = vocab.size
    bos_id = vocab.size + 1
    eos_id = vocab.size + 2
    seq_vocab_size = vocab.size + 3

    train_loader, valid_loader, test_loader = _build_loaders(cfg, vocab, tail_frames)

    model = LASGlobal(
        in_dim=feat_dim,
        vocab_size=seq_vocab_size,
        enc_hidden=cfg.lstm_hidden,
        enc_layers=cfg.lstm_layers,
        enc_bidirectional=cfg.bidirectional,
        dec_embed=cfg.las_dec_embed,
        dec_hidden=cfg.las_dec_hidden,
        dec_layers=cfg.las_dec_layers,
        attn_dim=cfg.las_attn_dim,
        dropout=cfg.dropout,
        enc_subsample=cfg.las_enc_subsample,
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")
    logger.info("Start training (Classic LAS: Global Attention)...")

    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        train_word_edits = 0
        train_word_ref = 0

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
                logits = model(feats, feat_lens, y_in)
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
                
                be, br = _batch_wer(pred, y_out, pad_id=pad_id, eos_id=eos_id, bos_id=bos_id)
                train_word_edits += be
                train_word_ref += br
                
                mask = y_out != pad_id
                if mask.any():
                    correct += int((pred[mask] == y_out[mask]).sum().item())
                    total += int(mask.sum().item())

            if it % cfg.log_every == 0:
                wer = float(train_word_edits / train_word_ref) if train_word_ref > 0 else 0.0
                acc = float(correct / total) if total > 0 else 0.0
                logger.info(
                    f"Epoch {ep:03d} | iter {it:05d} | train_loss {loss_meter.avg:.4f} "
                    f"| tok_acc {acc:.4f} | wer {wer:.4f} | n_tok {total}"
                )

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
            logger.info(
                f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} | tok_acc {val_res.tok_acc:.4f} | wer {val_res.wer:.4f} | n_words {val_res.n_words}"
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
    logger.info(f"[TEST] loss {test_res.mean_loss:.4f} | tok_acc {test_res.tok_acc:.4f} | wer {test_res.wer:.4f} | n_words {test_res.n_words}")

    save_json(cfg.run_dir / "result.json", {"best_val_loss": best_val, "test": asdict(test_res)})
    logger.info("Done.")
