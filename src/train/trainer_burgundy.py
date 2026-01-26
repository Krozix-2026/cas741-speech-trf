import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, DefaultDict, Optional
from collections import defaultdict
import warnings
import pickle
import numpy as np

from network.loss import DownWeightCompetitors
from network.RNN import LocalistRNN
from utils import compress_and_norm_64T, load_pickle_quiet
from speech_dataset.earshot_dataset import ContinuousStreamDataset




def pooled_tail_logits_for_chunk(logits_chunk,   # (B, Tc, V)
                                 metas,          # list of dicts with "marks"
                                 t0, t1,         # 本 chunk 在原序列中的 [t0, t1)
                                 device):
    """
    从一个时间 chunk 里抽取所有 token 的“末尾 k 帧平均 logit”
    返回:
      logits_tok: (N_tok, V)
      targets:    (N_tok,)
    """
    B, Tc, V = logits_chunk.shape

    logits_list = []
    target_list = []

    for b in range(B):
        marks = metas[b]["marks"]
        for (s, e, wid) in marks:
            # 与 chunk 无交集就跳过
            if e <= t0 or s >= t1:
                continue

            # 映射到 chunk 内局部坐标
            s_loc = max(0, s - t0)
            e_loc = min(Tc, e - t0)
            L = e_loc - s_loc
            if L <= 0:
                continue

            # 末尾 k 帧 (<=10)
            k = 10 if L >= 10 else L
            s_tail = e_loc - k

            pool = logits_chunk[b, s_tail:e_loc].mean(dim=0)  # (V,)
            logits_list.append(pool)
            target_list.append(int(wid))

    if not logits_list:
        return None, None

    logits_tok = torch.stack(logits_list, dim=0).to(device)  # (N_tok, V)
    targets = torch.tensor(target_list, device=device, dtype=torch.long)
    return logits_tok, targets

def read_mald_words_with_freq(txt_path: str) -> Tuple[List[str], Dict[str, int]]:
    words, freqs = [], {}
    for line in Path(txt_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        w, c = line.split()
        w = w.upper()
        words.append(w)
        freqs[w] = int(c)
    return words, freqs

def build_token_store(gt_root: str, speakers: List[str], words: List[str]):
    """
    Returns:
      bank: Dict[spk][word] -> List[np.ndarray (T,64)]
      words_present: sorted list of words across these speakers
      words_by_spk: Dict[spk] -> List[word]
    """
    root = Path(gt_root)
    bank: Dict[str, Dict[str, List[np.ndarray]]] = {}
    words_present = set()
    words = [w.upper() for w in words]

    for spk in speakers:
        spk_dir = root / spk
        spk_map: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        if not spk_dir.exists():
            print(f"[WARN] Missing speaker dir: {spk_dir}")
            bank[spk] = {}
            continue
        for w in words:
            p = spk_dir / f"{w}_{spk.upper()}.pickle"
            if not p.exists():
                continue
            try:
                arr = load_pickle_quiet(str(p))  # (64,T)
                # print("arr:", arr.shape)
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
                continue
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                continue
            if arr.shape[0] == 64:
                gt = arr
            elif arr.shape[1] == 64:
                gt = arr.T
            else:
                continue
            x = compress_and_norm_64T(gt)  # (T,64) float32
            # print("x:", x.shape)
            if x.shape[1] != 64 or x.shape[0] <= 0:
                continue
            spk_map[w].append(x)
            words_present.add(w)
        bank[spk] = dict(spk_map)

    words_present = sorted(words_present)
    words_by_spk: Dict[str, List[str]] = {
        spk: sorted([w for w, toks in bank.get(spk, {}).items() if len(toks) > 0])
        for spk in speakers
    }
    return bank, words_present, words_by_spk


def build_per_speaker_withheld(words_all, speakers, seed=0):
    rng = np.random.RandomState(seed)
    withheld = {}
    for spk in speakers:
        idx = np.arange(len(words_all))
        rng.shuffle(idx)
        k = max(1, len(idx)//16)
        withheld_words = set(words_all[i] for i in idx[:k])
        withheld[spk] = withheld_words
    return withheld

def stream_collate(batch):
    xs, ys, metas = zip(*batch)
    T_max = max(x.shape[0] for x in xs)
    B = len(xs)
    X = torch.zeros(B, T_max, xs[0].shape[1], dtype=torch.float32)
    V = ys[0].shape[1]
    Y = torch.zeros(B, T_max, V, dtype=torch.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        X[i, :T] = x
        Y[i, :T] = y
    return X, Y, list(metas)

# --------------------------
# Eval (last-100ms rule)
# --------------------------
@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    total_words, correct = 0, 0
    for X, _, metas in loader:
        X = X.to(device)
        logits, _ = model(X, None)
        # probs = torch.softmax(logits, dim=-1).cpu().numpy()   # (B,T,V)
        probs = torch.sigmoid(logits).cpu().numpy()   # (B,T,V)，如果你想用到的话

        B, T, V = probs.shape
        for i, meta in enumerate(metas):
            for (s, e, wid) in meta["marks"]:
                L = e - s
                if L <= 0:
                    continue
                k = 10 if L >= 10 else L
                e_clip = min(e, T)
                if e_clip <= 0:
                    continue
                s_clip = max(0, e_clip - k)
                avg = logits[i, s_clip:e_clip].mean(dim=0)
                pred = int(avg.argmax())
                correct += int(pred == wid)
                total_words += 1
    return correct / max(1, total_words)

@torch.no_grad()
def evaluate_concatenated(model, ds: 'ContinuousStreamDataset', N_segments=10, device="cuda"):
    Xs, marks_all = [], []
    t0 = 0
    for i in range(N_segments):
        X, Y, meta = ds[i]
        Xs.append(X.numpy())
        for (s,e,w) in meta["marks"]:
            marks_all.append((t0+s, t0+e, w))
        t0 += X.shape[0]
    big_X = np.concatenate(Xs, axis=0)  # (T_total, 64)

    model.eval()
    h = None
    chunk = 2000
    probs_all = []
    for t in range(0, big_X.shape[0], chunk):
        x = torch.from_numpy(big_X[t:t+chunk]).unsqueeze(0).to(device)
        logits, h = model(x, h)
        probs_all.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
    P = np.vstack(probs_all)  # (T_total, V)

    correct = 0
    for (s,e,w) in marks_all:
        k = min(10, e-s)
        avg = P[e-k:e].mean(axis=0)
        pred = int(avg.argmax())
        correct += int(pred == w)
    return correct / max(1, len(marks_all))


@torch.no_grad()
def eval_with_margin_rule_localist(model, loader, device, margin=0.05, dwell_frames=10):
    model.eval()
    correct, total = 0, 0
    for X, _, metas in loader:
        X = X.to(device)
        logits, _ = model(X, None)              # (B,T,V=vocab)
        probs = torch.sigmoid(logits) 

        B, T, V = probs.shape
        probs = probs.cpu()
        for b in range(B):
            P = probs[b].numpy()                # (T,V)
            for (s,e,wid) in metas[b]["marks"]:
                s = max(0, s); e = min(T, e)
                if e <= s: continue
                tgt = P[s:e, wid]
                max_others = np.max(P[s:e, :][:, np.arange(V) != wid], axis=1)
                margin_series = tgt - max_others
                good = (margin_series >= margin).astype(np.int32)
                run = 0; passed = False
                for g in good:
                    run = run + 1 if g else 0
                    if run >= dwell_frames:
                        passed = True; break
                correct += int(passed); total += 1
    return correct / max(1, total)

def check_label_consistency(ds: 'ContinuousStreamDataset'):
    X, Y, meta = ds[0]
    assert meta["marks"], "No tokens in first segment"
    wid0 = meta["marks"][0][2]
    inv_vocab = {i:w for w,i in ds.word2id.items()}
    word_str = inv_vocab[wid0]
    s,e,_ = meta["marks"][0]
    avg_hot = float(Y[s:e, wid0].float().mean().item())
    max_ones = float(Y[s:e, :].sum(dim=1).max().item())
    print(f"[Check] V={len(ds.words)}, wid={wid0}({word_str}), avg-hot={avg_hot:.3f}, per-frame max-ones={max_ones:.1f}")

# -------- tail mask (last-100ms frames only) --------
def build_tail_mask(metas, T_max, device):
    """
    metas: list of dicts (batch), each with 'marks'
    return: BoolTensor (B,T_max) True only on the last min(10,L) frames of each token
    """
    B = len(metas)
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for i, meta in enumerate(metas):
        for (s, e, _) in meta["marks"]:
            L = e - s
            if L <= 0: 
                continue
            k = 10 if L >= 10 else L
            e_clip = min(e, T_max)
            s_clip = max(0, e_clip - k)
            if s_clip < e_clip:
                mask[i, s_clip:e_clip] = True
    return mask

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size=512, warmup_steps=4000, last_epoch=-1):
        self.model_size = model_size
        self.warmup = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self._step_count, 1)
        scale = (self.model_size ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

def run_once(cfg, device):
    # vocab list
    words_all, freqs = read_mald_words_with_freq(cfg.neighbors)
    print(f"#Total MALD words read: {len(words_all)}")

    words_all = sorted(words_all, key=lambda w: -freqs[w])

    # 0) Build speakers & token store ONCE
    speakers_all = ["Agnes","Allison","Bruce","Junior","Princess","Samantha",
                    "Tom","Victoria","Alex","Ava","Fred","Kathy","Ralph","Susan","Vicki","MALD"]

    bank_all, present_all, words_by_spk_all = build_token_store(cfg.gammatone_dir, speakers_all, words_all)

    vocab_words = sorted(set(present_all))

    speakers_used = speakers_all
    bank_used = bank_all
    # keep only vocab_words entries per speaker
    words_by_spk_used = {
        spk: sorted(set(ws) & set(vocab_words))
        for spk, ws in words_by_spk_all.items()
        if spk in speakers_used
    }

    # 2) Build per-speaker withheld map ON THE ACTUAL vocab & speakers you’ll use
    withheld_map = build_per_speaker_withheld(vocab_words, speakers_used, seed=0)

    # 3) Datasets: SAME speakers; train excludes withheld_s; val uses only withheld_s
    train_ds = ContinuousStreamDataset(
        token_store=bank_used, speakers=speakers_used,
        vocab_words=vocab_words, words_by_spk=words_by_spk_used,
        train=True, segments_per_epoch=25*32,
        insert_silence=True, noise_snr_db=20.0,
        withheld_map=withheld_map, use_withheld_only=False
    )

    val_ds = ContinuousStreamDataset(
        token_store=bank_used, speakers=speakers_used,
        vocab_words=vocab_words, words_by_spk=words_by_spk_used,
        train=False, segments_per_epoch=25,
        insert_silence=False, noise_snr_db=None,
        withheld_map=withheld_map, use_withheld_only=True
    )



    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=stream_collate,
                            pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds,   batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=stream_collate,
                            pin_memory=True)


    train_eval_ds = ContinuousStreamDataset(
        token_store=bank_used, speakers=speakers_used,
        vocab_words=vocab_words, words_by_spk=words_by_spk_used,
        train=False,                       # no noise / no random silences
        segments_per_epoch=25,             # just a few segments for eval speed
        insert_silence=False, noise_snr_db=None,
        withheld_map=withheld_map, use_withheld_only=False  # <-- trained tokens
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=32, shuffle=False,
        num_workers=0, collate_fn=stream_collate, pin_memory=True
    )

    print(f"#Vocab: {len(vocab_words)} | Speakers used: {len(speakers_used)}")

    check_label_consistency(train_ds)

    V = len(vocab_words)
    model = LocalistRNN(input_dim=64, hidden=2048, layers=1, out_dim=V, dropout=0.1).to(device)

    # bce = nn.BCEWithLogitsLoss(reduction="mean")

    llill = DownWeightCompetitors(
        by=4.0,          # 或 5, 10; 表示 non-target loss 除以 c
        axis=-1,
        from_logits=True
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)  # 基础 lr=1.0，由 Noam 缩放
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=1e-4)

    best_acc = 0.0
    
    for ep in range(1, cfg.epoch + 1):
        if ep == 1:
            with torch.no_grad():
                Xb, Yb, _ = next(iter(train_loader))
                ratio = (Yb.sum(dim=-1) > 0).float().mean().item()
                print(f"[Sanity] fraction of word frames in a train batch: {ratio:.3f}")

        model.train()
        total_loss, total_frames = 0.0, 0

        
        CHUNK = 100   # 每个时间块长度（100=1s）。可试 50/100/200
        for X, Y, metas in train_loader:
            X = X.to(device)
            Y = Y.to(device) 
            # print("X:", X.shape)#[32, 1000, 64]

            B, T, _ = X.shape
            h = None
            for t0 in range(0, T, CHUNK):
                t1 = min(T, t0 + CHUNK)
                x_chunk = X[:, t0:t1, :]                     # (B,Tc,64)
                y_chunk = Y[:, t0:t1, :]        # (B, Tc, V)，包含词帧 & 静音帧(全0)

    
                logits_chunk, h = model(x_chunk, h)          # (B,Tc,V)

                logits_tok, targets = pooled_tail_logits_for_chunk(
                    logits_chunk, metas, t0, t1, device
                )

                # y_pred 需要有一个 “time” 维度，随便放一个长度为 1 的 time 维度：
                y_pred = logits_tok.unsqueeze(1)  # (N_tok, 1, V)

                # y_true 需要 (N_tok, 1, V+1)：one-hot + region mask=1
                Y_tok = torch.zeros_like(y_pred)[:, :, :V]  # (N_tok, 1, V)
                Y_tok[torch.arange(len(targets)), 0, targets] = 1.0
                region = torch.ones(len(targets), 1, 1, device=device)
                y_true_tok = torch.cat([Y_tok, region], dim=-1)  # (N_tok, 1, V+1)

                loss = llill(y_pred, y_true_tok)



                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                # detach 隐状态，避免跨块图连通
                if isinstance(h, tuple):
                    h = tuple(s.detach() for s in h)
                elif h is not None:
                    h = h.detach()

                total_loss += float(loss.item()) * logits_tok.shape[0]
                total_frames += logits_tok.shape[0]

        train_loss = total_loss / max(1, total_frames)

        train_acc = evaluate_loader(model, train_eval_loader, device)
        val_acc = evaluate_loader(model, val_loader, device)

        # wrong
        val_acc_stream = evaluate_concatenated(model, val_ds, N_segments=10, device=device)

        train_acc_margin = eval_with_margin_rule_localist(model, train_eval_loader, device,
                                             margin=0.05, dwell_frames=10)
        val_acc_margin = eval_with_margin_rule_localist(model, val_loader, device,
                                           margin=0.05, dwell_frames=10)


        scheduler.step()

        print(
            f"[Epoch {ep:02d}] loss {train_loss:.6f} | "
            f"TRAIN(trained tokens) acc {train_acc*100:.2f}% "
            f"(margin {train_acc_margin*100:.2f}%) | "
            f"VAL(new tokens) acc {val_acc*100:.2f}% "
            f"(margin {val_acc_margin*100:.2f}%)"
        )
        
        # 用 TBPTT 版本的 tail 帧准确率做个 sanity
        with torch.no_grad():
            X_dbg, _, M_dbg = next(iter(train_loader))
            X_dbg = X_dbg.to(device)
            B, T, _ = X_dbg.shape
            h = None
            preds_tok = []
            tgts_tok  = []
            for t0 in range(0, T, CHUNK):
                t1 = min(T, t0 + CHUNK)
                x_chunk = X_dbg[:, t0:t1, :]
                logits_chunk, h = model(x_chunk, h)
                # 复用 token_tail_loss_for_chunk 里的逻辑做“推理端 token 预测”
                # 只是不算 loss，而是取 argmax
                Bc, Tc, V = logits_chunk.shape
                for b in range(Bc):
                    marks = M_dbg[b]["marks"]
                    for (s, e, wid) in marks:
                        if e <= t0 or s >= t1:
                            continue
                        s_loc = max(0, s - t0)
                        e_loc = min(Tc, e - t0)
                        L = e_loc - s_loc
                        if L <= 0:
                            continue
                        k = 10 if L >= 10 else L
                        s_tail = e_loc - k
                        pool = logits_chunk[b, s_tail:e_loc].mean(dim=0)  # (V,)
                        preds_tok.append(int(pool.argmax().item()))
                        tgts_tok.append(int(wid))
                if isinstance(h, tuple):
                    h = tuple(s.detach() for s in h)
                elif h is not None:
                    h = h.detach()
            if len(tgts_tok) > 0:
                acc_tok = (torch.tensor(preds_tok) == torch.tensor(tgts_tok)).float().mean().item()
                uniq_pred = len(set(preds_tok))
                print(f"[Sanity] token-tail-acc={acc_tok:.3f}, token-uniq-pred={uniq_pred}")


        if ep > 400 and val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "vocab_words": vocab_words},
                cfg.ckpt_dir / "RNN_ori_localist_{ep}.pt"
            )

    print(f"Best val acc: {best_acc*100:.2f}%")
