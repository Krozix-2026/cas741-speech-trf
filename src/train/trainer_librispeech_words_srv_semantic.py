# train/trainer_librispeech_words_srv_semantic.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.schema import TrainConfig
from network.lstm_frame_srv_semantic import LSTMFrameSRVSemantic

from utils.utils import (
    AverageMeter,
    count_parameters,
    save_json,
    set_seed,
    setup_logger,
)
from utils.srv import make_srv_table
from speech_dataset.librispeech_aligned_words import (
    build_word_vocab_from_manifest,
    LibriSpeechAlignedWords,
)
from speech_dataset.collate_aligned_words import collate_aligned_words

IGNORE_INDEX = -100


@dataclass
class EvalResult:
    mean_loss: float
    mean_cos_word: float
    mean_cos_sem: float
    sup_frames: int
    sup_words: int


def _build_loaders(cfg: TrainConfig, vocab, tail_frames: int):
    if cfg.librispeech_manifest is None:
        raise ValueError("TrainConfig.librispeech_manifest must be set.")
    manifest_path = Path(cfg.librispeech_manifest)

    train_ds = LibriSpeechAlignedWords(manifest_path, cfg.train_subset, vocab, tail_frames, cfg.limit_train_samples)
    valid_ds = LibriSpeechAlignedWords(manifest_path, cfg.valid_subset, vocab, tail_frames, cfg.limit_valid_samples)
    test_ds  = LibriSpeechAlignedWords(manifest_path, cfg.test_subset,  vocab, tail_frames, cfg.limit_test_samples)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_aligned_words)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_aligned_words)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_aligned_words)
    return train_loader, valid_loader, test_loader


def _save_ckpt(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "epoch": epoch, "best_val": best_val},
        path,
    )


def _srv_loss_and_cos_frame(
    y: torch.Tensor, # (B,T,D)
    targets: torch.Tensor, # (B,T) wid or IGNORE
    srv_table: torch.Tensor, # (V,D)
    loss_type: str,
) -> tuple[torch.Tensor, float, int]:
    mask = targets != IGNORE_INDEX
    if not mask.any():
        return y.sum() * 0.0, 0.0, 0

    y_lab = y[mask] # (N,D)
    wid = targets[mask].long() # (N,)
    t_lab = srv_table[wid] # (N,D)

    y_n = F.normalize(y_lab.float(), dim=-1, eps=1e-8)
    t_n = t_lab.float()
    cos = (y_n * t_n).sum(dim=-1)
    mean_cos = float(cos.mean().item())

    if loss_type.lower() == "cosine":
        loss = (1.0 - cos).mean()
    elif loss_type.lower() == "mse":
        loss = F.mse_loss(y_n, t_n)
    else:
        raise ValueError(f"Unknown srv_loss: {loss_type}")

    return loss, mean_cos, int(wid.numel())


def _srv_loss_and_cos_word(
    pred: torch.Tensor, # (B,W,D) semantic predictions (for each position i)
    next_ids: torch.Tensor, # (B,W) wid (target for each position i) or IGNORE
    srv_table: torch.Tensor, # (V,D)
    loss_type: str,
) -> tuple[torch.Tensor, float, int]:
    mask = next_ids != IGNORE_INDEX
    if not mask.any():
        return pred.sum() * 0.0, 0.0, 0

    p = pred[mask] # (N,D)
    wid = next_ids[mask].long()# (N,)
    t = srv_table[wid] # (N,D)

    p_n = F.normalize(p.float(), dim=-1, eps=1e-8)
    t_n = t.float()
    cos = (p_n * t_n).sum(dim=-1)
    mean_cos = float(cos.mean().item())

    if loss_type.lower() == "cosine":
        loss = (1.0 - cos).mean()
    elif loss_type.lower() == "mse":
        loss = F.mse_loss(p_n, t_n)
    else:
        raise ValueError(f"Unknown srv_loss: {loss_type}")

    return loss, mean_cos, int(wid.numel())


@torch.no_grad()
def _run_eval(model, loader, device, srv_table, cfg, tail_frames: int) -> EvalResult:
    model.eval()
    loss_meter = AverageMeter()
    cos_word_meter = AverageMeter()
    cos_sem_meter = AverageMeter()
    sup_frames = 0
    sup_words = 0

    shift = int(getattr(cfg, "semantic_shift", 1))
    alpha = float(getattr(cfg, "semantic_alpha", 1.0))
    beta = float(getattr(cfg, "semantic_beta", 0.5))

    for batch in loader:
        feats = batch.feats.to(device, non_blocking=True)
        feat_lens = batch.feat_lens.to(device, non_blocking=True)
        targets = batch.targets.to(device, non_blocking=True)

        word_starts = batch.word_starts.to(device, non_blocking=True)
        word_ends = batch.word_ends.to(device, non_blocking=True)
        word_lens = batch.word_lens.to(device, non_blocking=True)
        word_ids = batch.word_ids.to(device, non_blocking=True)

        y_frame, out_lens, pred_sem = model(
            feats, feat_lens,
            word_starts=word_starts, word_ends=word_ends, word_lens=word_lens,
            tail_frames=tail_frames,
            detach_frame_for_sem=bool(getattr(cfg, "semantic_detach_frame", False)),
        )

        # frame-level word SRV
        loss_word, cos_word, n_frames = _srv_loss_and_cos_frame(y_frame, targets, srv_table, cfg.srv_loss)

        # semantic next-word SRV
        W = int(word_ids.shape[1])
        if W <= shift:
            loss_sem = loss_word * 0.0
            cos_sem = 0.0
            n_words = 0
        else:
            # predict word_{i+shift} from state at position i
            next_ids = word_ids[:, shift:]# (B, W-shift)
            pred_use = pred_sem[:, :W-shift, :] # (B, W-shift, D)
            loss_sem, cos_sem, n_words = _srv_loss_and_cos_word(pred_use, next_ids, srv_table, cfg.srv_loss)

        loss = alpha * loss_word + beta * loss_sem

        loss_meter.update(float(loss.item()), n=feats.size(0))
        cos_word_meter.update(cos_word, n=max(1, n_frames))
        cos_sem_meter.update(cos_sem, n=max(1, n_words))
        sup_frames += n_frames
        sup_words += n_words

    return EvalResult(
        mean_loss=loss_meter.avg,
        mean_cos_word=cos_word_meter.avg,
        mean_cos_sem=cos_sem_meter.avg,
        sup_frames=sup_frames,
        sup_words=sup_words,
    )


def run_once(cfg: TrainConfig, device: str) -> None:
    cfg.ensure_dirs()
    logger = setup_logger(cfg.log_dir, name=cfg.run_id())
    save_json(cfg.run_dir / "config.json", cfg)

    set_seed(cfg.seed)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {cfg.run_dir}")

    env_sr = int(getattr(cfg, "env_sr", 100))
    tail_ms = int(getattr(cfg, "tail_ms", 100))
    topk = int(getattr(cfg, "word_vocab_topk", 20000))
    feat_dim = int(getattr(cfg, "coch_feat_dim", 64))

    tail_frames = max(1, int(round((tail_ms / 1000.0) * env_sr)))
    logger.info(f"Word supervision tail: {tail_ms} ms => {tail_frames} frames at env_sr={env_sr}")

    manifest_path = Path(cfg.librispeech_manifest)
    logger.info(f"Manifest: {manifest_path}")

    vocab = build_word_vocab_from_manifest(manifest_path, topk=topk)
    logger.info(f"Word vocab size: {vocab.size} (unk_id={vocab.unk_id}) topk={topk}")

    train_loader, valid_loader, test_loader = _build_loaders(cfg, vocab, tail_frames)

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
    model = LSTMFrameSRVSemantic(
        in_dim=feat_dim,
        srv_dim=D,
        frame_hidden=cfg.lstm_hidden,
        frame_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        word_rep_dim=int(getattr(cfg, "word_rep_dim", 256)),
        word_lstm_hidden=int(getattr(cfg, "word_lstm_hidden", 256)),
        word_lstm_layers=int(getattr(cfg, "word_lstm_layers", 1)),
        rep_dropout=float(getattr(cfg, "rep_dropout", 0.1)),
        word_dropout_p=float(getattr(cfg, "word_dropout_p", 0.25)),
        rep_noise_std=float(getattr(cfg, "rep_noise_std", 0.0)),
    ).to(device)
    logger.info(f"Model params: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.startswith("cuda")))

    best_val = float("inf")
    alpha = float(getattr(cfg, "semantic_alpha", 1.0))
    beta = float(getattr(cfg, "semantic_beta", 0.5))
    shift = int(getattr(cfg, "semantic_shift", 1))

    logger.info(
        f"Start training (Stage C): alpha(word_tail)={alpha} beta(semantic)={beta} shift={shift} "
        f"word_dropout_p={getattr(cfg,'word_dropout_p',0.25)} detach_frame={getattr(cfg,'semantic_detach_frame',False)}"
    )

    for ep in range(1, cfg.epoch + 1):
        model.train()
        loss_meter = AverageMeter()
        cos_word_meter = AverageMeter()
        cos_sem_meter = AverageMeter()
        sup_frames = 0
        sup_words = 0

        for it, batch in enumerate(train_loader, start=1):
            feats = batch.feats.to(device, non_blocking=True)
            # print("feats:", feats.shape)#[32, 1698, 64]
            feat_lens = batch.feat_lens.to(device, non_blocking=True)
            # print("feat_lens:", feat_lens)#[1698, 1611, 1590,...
            targets = batch.targets.to(device, non_blocking=True)
            # print("targets:", targets.shape)#[32, 1698]

            word_starts = batch.word_starts.to(device, non_blocking=True)
            # print("word_starts:", word_starts)#[  20,   40,   58,  ...
            word_ends = batch.word_ends.to(device, non_blocking=True)
            # print("word_ends:", word_ends)#[[  40,   58,   75,  ...,
            word_lens = batch.word_lens.to(device, non_blocking=True)
            # print("word_lens:", word_lens)#[48, 42, 42, 43,...
            word_ids = batch.word_ids.to(device, non_blocking=True)
            # print("word_ids:", word_ids)#[[  17,   44,   71,  ...

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and device.startswith("cuda"))):
                y_frame, out_lens, pred_sem = model(
                    feats, feat_lens,
                    word_starts=word_starts, word_ends=word_ends, word_lens=word_lens,
                    tail_frames=tail_frames,
                    detach_frame_for_sem=bool(getattr(cfg, "semantic_detach_frame", False)),
                )

                loss_word, cos_word, n_frames = _srv_loss_and_cos_frame(y_frame, targets, srv_table, cfg.srv_loss)

                W = int(word_ids.shape[1])
                if W <= shift:
                    loss_sem = loss_word * 0.0
                    cos_sem = 0.0
                    n_words = 0
                else:
                    next_ids = word_ids[:, shift:] # (B, W-shift)
                    pred_use = pred_sem[:, :W-shift, :]  #(B, W-shift, D)
                    loss_sem, cos_sem, n_words = _srv_loss_and_cos_word(pred_use, next_ids, srv_table, cfg.srv_loss)

                loss = alpha * loss_word + beta * loss_sem

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feats.size(0))
            cos_word_meter.update(cos_word, n=max(1, n_frames))
            cos_sem_meter.update(cos_sem, n=max(1, n_words))
            sup_frames += n_frames
            sup_words += n_words

            if it % cfg.log_every == 0:
                logger.info(
                    f"Epoch {ep:03d} | iter {it:05d} | loss {loss_meter.avg:.4f} "
                    f"| cos_word {cos_word_meter.avg:.4f} | cos_sem {cos_sem_meter.avg:.4f} "
                    f"| sup_frames {sup_frames} | sup_words {sup_words}"
                )

        if (ep % cfg.eval_every_epochs) == 0:
            val_res = _run_eval(model, valid_loader, device, srv_table, cfg, tail_frames)
            logger.info(
                f"[VALID] Epoch {ep:03d} | loss {val_res.mean_loss:.4f} "
                f"| cos_word {val_res.mean_cos_word:.4f} | cos_sem {val_res.mean_cos_sem:.4f} "
                f"| sup_frames {val_res.sup_frames} | sup_words {val_res.sup_words}"
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

    test_res = _run_eval(model, test_loader, device, srv_table, cfg, tail_frames)
    logger.info(
        f"[TEST] loss {test_res.mean_loss:.4f} | cos_word {test_res.mean_cos_word:.4f} "
        f"| cos_sem {test_res.mean_cos_sem:.4f} | sup_frames {test_res.sup_frames} | sup_words {test_res.sup_words}"
    )
    save_json(cfg.run_dir / "result.json", {"best_val_loss": best_val, "test": asdict(test_res)})
    logger.info("Done.")