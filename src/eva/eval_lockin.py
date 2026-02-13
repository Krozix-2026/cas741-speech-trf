# eval_lockin.py
# ------------------------------------------------------------
# Evaluate "lexical lock-in" dynamics for word-aligned LSTM model.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Make imports robust when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from network.lstm_frame_classifier import LSTMFrameClassifier  # noqa

# Try to reuse the EXACT vocab builder used in training (recommended)
try:
    from speech_dataset.librispeech_aligned_words import build_word_vocab_from_manifest  # noqa
except Exception:
    build_word_vocab_from_manifest = None


@dataclass
class TokenMetric:
    utt_id: str
    word: str
    wid: int
    sf: int
    ef: int
    dur_frames: int
    dur_ms: float
    lockin_ms: float  # -1 if not found
    end_correct: int  # 1/0
    unk: int          # 1/0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default=r"C:\Dataset\LibriSpeech\manifest_librispeech_coch_align.jsonl",
                   help="Path to manifest_librispeech_coch_align.jsonl")
    p.add_argument("--ckpt", type=str,
                   default=r"C:\linux_project\CAS741\cas741-speech-trf\runs\librispeech_LSTM_WORD_baseline_s000\ckpt\best.pt",
                   help="Path to best.pt (or last.pt)")
    p.add_argument("--subset", type=str, default="dev-clean",
                   choices=["train-clean-100", "dev-clean", "test-clean"])
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output dir (default: alongside ckpt -> ../eval_lockin_SUBSET)")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda|cpu (auto fallback if cuda unavailable)")
    p.add_argument("--topk", type=int, default=20000,
                   help="Word vocab topK (must match training)")
    p.add_argument("--env_sr", type=int, default=100,
                   help="Cochleagram env sample rate in Hz (frames/s)")
    p.add_argument("--feat_dim", type=int, default=64,
                   help="Cochleagram feature dim (channels)")
    p.add_argument("--hidden", type=int, default=None,
                   help="Override hidden size (default from config.json if found)")
    p.add_argument("--layers", type=int, default=None,
                   help="Override layers (default from config.json if found)")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override dropout (default from config.json if found)")
    p.add_argument("--max_utts", type=int, default=None,
                   help="Limit number of utterances (debug/speed)")
    p.add_argument("--skip_unk", action="store_true",
                   help="Skip <unk> targets in evaluation")
    p.add_argument("--min_word_frames", type=int, default=5,
                   help="Skip words shorter than this (in frames)")
    p.add_argument("--delta", type=float, default=1.0,
                   help="Margin threshold for lock-in: logit(correct)-logit(comp) > delta")
    p.add_argument("--m", type=int, default=5,
                   help="Consecutive frames required for lock-in stability")
    p.add_argument("--delays_ms", type=str, default="50,100,200,300",
                   help="Comma-separated early-eval delays in ms")
    p.add_argument("--curve_bins", type=int, default=60,
                   help="Bins for mean margin curve (relative position 0..1)")

    # ===== CHANGED: controls for example competition plots =====
    p.add_argument("--plot_examples", type=int, default=5,
                   help="Plot N example word curves (competition + margin). 0 to disable.")
    p.add_argument("--example_seed", type=int, default=0,
                   help="Random seed for sampling examples.")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_config_near_ckpt(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    try:
        ckpt_dir = ckpt_path.parent
        run_dir = ckpt_dir.parent
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            return read_json(cfg_path)
    except Exception:
        pass
    return None


def build_vocab(manifest_path: Path, topk: int) -> Tuple[Dict[str, int], List[str], int]:
    if build_word_vocab_from_manifest is None:
        from collections import Counter
        cnt = Counter()
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                for w, sf, ef in obj.get("words", []):
                    cnt[w] += 1
        itos = ["<unk>"] + [w for w, _ in cnt.most_common(topk)]
        stoi = {w: i for i, w in enumerate(itos)}
        return stoi, itos, 0
    else:
        vocab = build_word_vocab_from_manifest(manifest_path, topk=topk)
        return vocab.stoi, vocab.itos, vocab.unk_id


def load_model(
    ckpt_path: Path,
    vocab_size: int,
    feat_dim: int,
    device: torch.device,
    hidden: Optional[int] = None,
    layers: Optional[int] = None,
    dropout: Optional[float] = None,
) -> LSTMFrameClassifier:
    cfg = load_config_near_ckpt(ckpt_path) or {}
    h = int(hidden if hidden is not None else cfg.get("lstm_hidden", 512))
    L = int(layers if layers is not None else cfg.get("lstm_layers", 3))
    d = float(dropout if dropout is not None else cfg.get("dropout", 0.1))

    model = LSTMFrameClassifier(
        in_dim=feat_dim,
        vocab_size=vocab_size,
        hidden=h,
        layers=L,
        dropout=d,
        bidirectional=False,  # enforced
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=str(device))
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_manifest_subset(manifest_path: Path, subset: str, max_utts: Optional[int]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("subset") != subset:
                continue
            items.append(obj)
            if max_utts is not None and len(items) >= int(max_utts):
                break
    return items


def lockin_time_from_margin(margin: torch.Tensor, delta: float, m: int) -> int:
    L = int(margin.numel())
    if L < m:
        return -1
    good = (margin > delta).to(torch.float32).view(1, 1, L)  # (1,1,L)
    kernel = torch.ones((1, 1, m), dtype=torch.float32, device=margin.device)
    sums = F.conv1d(good, kernel, stride=1)  # (1,1,L-m+1)
    idx = (sums[0, 0] >= float(m) - 1e-6).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return -1
    return int(idx[0].item())


def update_mean_curve(accum: np.ndarray, count: np.ndarray, y: np.ndarray, bins: int) -> None:
    L = len(y)
    if L < 2:
        return
    x_src = np.linspace(0.0, 1.0, num=L, dtype=np.float32)
    x_tgt = np.linspace(0.0, 1.0, num=bins, dtype=np.float32)
    y_rs = np.interp(x_tgt, x_src, y).astype(np.float32)
    accum += y_rs
    count += 1


def plot_and_save(fig_path: Path, kind: str, data: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt  # noqa

    ensure_dir(fig_path.parent)

    if kind == "cdf":
        xs = data["xs"]
        ys = data["ys"]
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Lock-in time (ms)")
        plt.ylabel("CDF (fraction locked-in)")
        plt.title("Lock-in time CDF")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

    elif kind == "mean_margin":
        x = data["x"]
        y = data["y"]
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Relative position in word (0=start, 1=end)")
        plt.ylabel("Mean margin (logit_correct - logit_best_comp)")
        plt.title("Mean margin curve (aligned by word onset)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

    elif kind == "examples":
        examples = data["examples"]

        # 1) probability curves: p(correct) vs p(best competitor)
        plt.figure(figsize=(9, 5))
        for ex in examples:
            plt.plot(ex["t_ms"], ex["p_correct"], alpha=0.9)
            plt.plot(ex["t_ms"], ex["p_comp"], alpha=0.9, linestyle="--")
        plt.xlabel("Time from word onset (ms)")
        plt.ylabel("Probability")
        plt.title("Competition: p(correct) (solid) vs p(best competitor) (dashed)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()

        # 2) margin curves with legend
        fig_path2 = fig_path.with_name("example_margin_curves.png")
        plt.figure(figsize=(9, 5))
        for ex in examples:
            plt.plot(ex["t_ms"], ex["margin"], alpha=0.9, label=ex["label"])
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Time from word onset (ms)")
        plt.ylabel("Margin (logit_correct - logit_best_comp)")
        plt.title("Margin curves (with competitor label)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(fig_path2, dpi=160)
        plt.close()

    else:
        raise ValueError(f"Unknown plot kind: {kind}")


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.ckpt)

    delays = [int(x.strip()) for x in args.delays_ms.split(",") if x.strip()]
    delays = sorted(list(set(delays)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # Output dir
    if args.out_dir is None:
        out_dir = ckpt_path.parent.parent / f"eval_lockin_{args.subset}"
    else:
        out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[Device] {device}")
    print(f"[Subset] {args.subset}")
    print(f"[Out] {out_dir}")

    # Vocab must match training
    stoi, itos, unk_id = build_vocab(manifest_path, topk=int(args.topk))
    vocab_size = len(itos)
    print(f"[Vocab] size={vocab_size} unk_id={unk_id} topk={args.topk}")

    # Model
    model = load_model(
        ckpt_path=ckpt_path,
        vocab_size=vocab_size,
        feat_dim=int(args.feat_dim),
        device=device,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    )
    print("[Model] loaded.")

    # Manifest items
    items = load_manifest_subset(manifest_path, args.subset, args.max_utts)
    print(f"[Manifest] {len(items)} utterances loaded.")

    # Aggregates
    lockin_ms_all: List[float] = []
    token_rows: List[TokenMetric] = []

    bins = int(args.curve_bins)
    mean_margin_accum = np.zeros((bins,), dtype=np.float64)
    mean_margin_count = np.zeros((bins,), dtype=np.float64)

    # ===== CHANGED: examples sampling =====
    want_examples = int(args.plot_examples)
    rng = np.random.default_rng(int(args.example_seed))
    ex_curves: List[Dict[str, Any]] = []

    n_utts = 0
    n_tokens = 0
    n_skipped_unk = 0
    n_short = 0

    for obj in items:
        utt_id = obj["utt_id"]
        coch_path = Path(obj["coch_path"])
        words = obj.get("words", [])

        if not coch_path.exists():
            print(f"[Skip] missing coch: {coch_path}")
            continue

        arr = np.load(str(coch_path))
        if arr.ndim != 2 or arr.shape[0] != int(args.feat_dim):
            print(f"[Skip] bad shape {arr.shape} for {coch_path}")
            continue

        T = int(arr.shape[1])
        x = torch.from_numpy(arr.astype(np.float32)).to(device)  # (F, T)
        x = x.transpose(0, 1).unsqueeze(0)  # (1, T, F)
        x_lens = torch.tensor([T], dtype=torch.long, device=device)

        with torch.inference_mode():
            logits_bt, out_lens = model(x, x_lens)  # (1,T,V)

        logits = logits_bt[0]  # (T,V)

        # Precompute top1/top2 for competitor margin
        top2_val, top2_idx = torch.topk(logits, k=2, dim=-1)  # (T,2)
        top1_idx = top2_idx[:, 0]
        top1_val = top2_val[:, 0]
        top2_val2 = top2_val[:, 1]

        # log-softmax only if we will sample examples
        logp = None
        if want_examples > 0:
            logp = torch.log_softmax(logits, dim=-1)

        for w, sf, ef in words:
            sf = int(sf)
            ef = int(ef)
            if ef <= sf:
                continue
            if sf < 0:
                sf = 0
            if ef > T:
                ef = T
            dur = ef - sf
            if dur < int(args.min_word_frames):
                n_short += 1
                continue

            wid = stoi.get(w, unk_id)
            is_unk = 1 if wid == unk_id else 0
            if args.skip_unk and is_unk:
                n_skipped_unk += 1
                continue

            # competitor logit per frame: if top1==wid -> competitor is top2 else top1
            comp_val = torch.where(top1_idx == wid, top2_val2, top1_val)  # (T,)
            margin = logits[:, wid] - comp_val  # (T,)
            margin_seg = margin[sf:ef]  # (dur,)

            lock_idx = lockin_time_from_margin(margin_seg, delta=float(args.delta), m=int(args.m))
            lock_ms = -1.0
            if lock_idx >= 0:
                lock_ms = (lock_idx / float(args.env_sr)) * 1000.0
                lockin_ms_all.append(lock_ms)

            end_pred = int(top1_idx[ef - 1].item())
            end_correct = 1 if end_pred == wid else 0

            tm = TokenMetric(
                utt_id=utt_id,
                word=w,
                wid=int(wid),
                sf=sf,
                ef=ef,
                dur_frames=dur,
                dur_ms=(dur / float(args.env_sr)) * 1000.0,
                lockin_ms=float(lock_ms),
                end_correct=int(end_correct),
                unk=int(is_unk),
            )

            # early accuracies at delays
            for dms in delays:
                t = sf + int(round((dms / 1000.0) * float(args.env_sr)))
                if t < ef:
                    pred = int(top1_idx[t].item())
                    tm.__dict__[f"acc_at_{dms}ms"] = 1 if pred == wid else 0
                else:
                    tm.__dict__[f"acc_at_{dms}ms"] = ""

            token_rows.append(tm)
            n_tokens += 1

            update_mean_curve(
                accum=mean_margin_accum,
                count=mean_margin_count,
                y=margin_seg.detach().to("cpu").numpy().astype(np.float32),
                bins=bins,
            )

            # ===== CHANGED: collect competition examples (ONLY ONE append path) =====
            if want_examples > 0 and logp is not None and len(ex_curves) < want_examples:
                # sample tokens stochastically rather than always early tokens in file
                # this avoids bias (e.g., always capturing first few words)
                # accept with prob p so we gradually fill the list
                # (simple scheme: accept if random < 0.02, fallback to fill)
                accept = (rng.random() < 0.02) or (len(ex_curves) < max(1, want_examples // 3))
                if accept:
                    seg_logp = logp[sf:ef, :]  # (dur, V)

                    # p(correct)
                    p_correct = torch.exp(seg_logp[:, wid]).detach().cpu().numpy().astype(np.float32)

                    # best competitor per frame (exclude wid)
                    seg_logp2 = seg_logp.clone()
                    seg_logp2[:, wid] = -1e9
                    comp_wid = torch.argmax(seg_logp2, dim=-1)  # (dur,)
                    p_comp = torch.exp(
                        seg_logp2.gather(1, comp_wid[:, None]).squeeze(1)
                    ).detach().cpu().numpy().astype(np.float32)

                    # margin in logits space (same competitor definition)
                    seg_logits = logits[sf:ef, :]
                    seg_logits2 = seg_logits.clone()
                    seg_logits2[:, wid] = -1e9
                    comp_wid_logits = torch.argmax(seg_logits2, dim=-1)
                    comp_logit = seg_logits2.gather(1, comp_wid_logits[:, None]).squeeze(1)
                    margin_local = (seg_logits[:, wid] - comp_logit).detach().cpu().numpy().astype(np.float32)

                    # dominant competitor id (mode)
                    comp_ids = comp_wid.detach().cpu().numpy()
                    comp_dom = int(np.bincount(comp_ids).argmax()) if comp_ids.size > 0 else -1
                    comp_word = itos[comp_dom] if 0 <= comp_dom < len(itos) else "<na>"

                    t_ms = (np.arange(dur, dtype=np.float32) / float(args.env_sr)) * 1000.0
                    ex_curves.append({
                        "t_ms": t_ms,
                        "p_correct": p_correct,
                        "p_comp": p_comp,
                        "margin": margin_local,
                        "label": f"{w} vs {comp_word} | {utt_id}",
                    })

        n_utts += 1
        if n_utts % 200 == 0:
            print(f"[OK] processed {n_utts} utts, tokens={n_tokens}")

    # ---- Save per-token CSV ----
    csv_path = out_dir / "token_metrics.csv"
    fields = [
        "utt_id", "word", "wid", "sf", "ef",
        "dur_frames", "dur_ms", "lockin_ms",
        "end_correct", "unk",
    ] + [f"acc_at_{d}ms" for d in delays]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        for tm in token_rows:
            row = {
                "utt_id": tm.utt_id,
                "word": tm.word,
                "wid": tm.wid,
                "sf": tm.sf,
                "ef": tm.ef,
                "dur_frames": tm.dur_frames,
                "dur_ms": f"{tm.dur_ms:.3f}",
                "lockin_ms": f"{tm.lockin_ms:.3f}",
                "end_correct": tm.end_correct,
                "unk": tm.unk,
            }
            for dms in delays:
                row[f"acc_at_{dms}ms"] = tm.__dict__.get(f"acc_at_{dms}ms", "")
            wcsv.writerow(row)

    print(f"[Save] {csv_path}")

    # ---- Summary stats ----
    end_acc = float(sum(tm.end_correct for tm in token_rows) / max(1, len(token_rows)))
    lockin_found = [x for x in lockin_ms_all if x >= 0]
    lockin_rate = float(len(lockin_found) / max(1, len(token_rows)))

    def percentile(xs: List[float], q: float) -> float:
        if not xs:
            return float("nan")
        xs2 = sorted(xs)
        k = (len(xs2) - 1) * q
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return float(xs2[lo])
        return float(xs2[lo] * (hi - k) + xs2[hi] * (k - lo))

    summary = {
        "subset": args.subset,
        "n_utts": n_utts,
        "n_tokens": len(token_rows),
        "skipped_unk": int(n_skipped_unk),
        "skipped_short": int(n_short),
        "env_sr": int(args.env_sr),
        "delta": float(args.delta),
        "m": int(args.m),
        "end_acc": end_acc,
        "lockin_rate": lockin_rate,
        "lockin_ms_median": percentile(lockin_found, 0.50),
        "lockin_ms_p80": percentile(lockin_found, 0.80),
        "lockin_ms_p90": percentile(lockin_found, 0.90),
    }

    for dms in delays:
        vals = []
        for tm in token_rows:
            v = tm.__dict__.get(f"acc_at_{dms}ms", "")
            if v == "" or v is None:
                continue
            vals.append(int(v))
        summary[f"acc_at_{dms}ms"] = float(sum(vals) / max(1, len(vals))) if vals else float("nan")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Save] {summary_path}")
    print("[Summary]", json.dumps(summary, indent=2))

    # ---- Plots ----
    if lockin_found:
        xs = np.array(sorted(lockin_found), dtype=np.float32)
        ys = np.arange(1, len(xs) + 1, dtype=np.float32) / float(len(xs))
        plot_and_save(out_dir / "lockin_cdf.png", "cdf", {"xs": xs, "ys": ys})
        print(f"[Plot] {out_dir / 'lockin_cdf.png'}")

    valid_mask = mean_margin_count > 0
    mean_curve = np.zeros_like(mean_margin_accum, dtype=np.float32)
    mean_curve[valid_mask] = (mean_margin_accum[valid_mask] / mean_margin_count[valid_mask]).astype(np.float32)
    x_rel = np.linspace(0.0, 1.0, num=bins, dtype=np.float32)
    plot_and_save(out_dir / "mean_margin_curve.png", "mean_margin", {"x": x_rel, "y": mean_curve})
    print(f"[Plot] {out_dir / 'mean_margin_curve.png'}")

    # ===== CHANGED: examples always have p_comp + margin =====
    if ex_curves:
        plot_and_save(out_dir / "example_competition_curves.png", "examples", {"examples": ex_curves})
        print(f"[Plot] {out_dir / 'example_competition_curves.png'}")
        print(f"[Plot] {out_dir / 'example_margin_curves.png'}")


if __name__ == "__main__":
    main()
