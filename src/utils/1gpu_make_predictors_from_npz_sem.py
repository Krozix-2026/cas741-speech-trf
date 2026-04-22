# 1gpu_make_predictors_from_npz_sem.py
# ------------------------------------------------------------
# Make TRF predictors from Appleseed gammatone100 npz using a
# 2-layer LSTMFrameSRVSemantic model:
#   - Layer1 (frame LSTM): continuous-time predictors over frames
#   - Layer2 (word LSTM): event-like predictors at word boundaries
#
# Inputs:
#   - gammatone100 npz: x (T,64), t0, tstep
#   - TextGrid: "segment 1.TextGrid" ... "segment 11b.TextGrid"
#
# Outputs (default two files per stimulus):
#   1) <stem>-lstm_frame_predictors.npz
#      - frame_h_mag, frame_h_chg (T,)
#      - [opt] frame_h (T,Hf) float16
#   2) <stem>-lstm_word_predictors.npz
#      - sem_c_mag_imp, sem_c_chg_imp (T,)  impulse at word events
#      - [opt] sem_c_impulse (T,Hw) float16 sparse
#      - word_starts, word_ends, sem_event_idx (W,), n_words
#      - [opt] sem_c_words (W,Hw) float16
#      - [opt] debug words (word_text, word_xmin_s, word_xmax_s)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch

#add project root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv_semantic import LSTMFrameSRVSemantic


# TextGrid parsing (Praat long text format)
_NAME_PAT = re.compile(r'^\s*name\s*=\s*"([^"]*)"\s*$')
_XMIN_PAT = re.compile(r'^\s*xmin\s*=\s*([0-9.+-eE]+)\s*$')
_XMAX_PAT = re.compile(r'^\s*xmax\s*=\s*([0-9.+-eE]+)\s*$')
_TEXT_PAT = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')


def parse_textgrid_words(textgrid_path: Path, tier_name: str = "words") -> List[Tuple[float, float, str]]:
    """
    Return list of (xmin_sec, xmax_sec, text) for non-empty intervals in the given tier.
    """
    lines = textgrid_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    in_target_tier = False
    cur_xmin = None
    cur_xmax = None
    cur_text = None
    intervals: List[Tuple[float, float, str]] = []

    for ln in lines:
        m = _NAME_PAT.match(ln)
        if m:
            nm = m.group(1)
            in_target_tier = (nm == tier_name)
            cur_xmin = cur_xmax = cur_text = None
            continue

        if not in_target_tier:
            continue

        m = _XMIN_PAT.match(ln)
        if m:
            cur_xmin = float(m.group(1))
            continue

        m = _XMAX_PAT.match(ln)
        if m:
            cur_xmax = float(m.group(1))
            continue

        m = _TEXT_PAT.match(ln)
        if m:
            cur_text = m.group(1)
            if cur_xmin is not None and cur_xmax is not None:
                txt = (cur_text or "").strip()
                if txt != "":
                    intervals.append((cur_xmin, cur_xmax, txt))
            cur_xmin = cur_xmax = cur_text = None
            continue

    return intervals


def intervals_to_frames(
    intervals: List[Tuple[float, float, str]],
    t0: float,
    tstep: float,
    T: int,
    mode: str = "end", # "start" | "end" | "center"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (xmin,xmax) seconds -> (start_frame, end_frame) frames (end exclusive),
    plus an event index per word according to mode.
    """
    ws: List[int] = []
    we: List[int] = []
    ev: List[int] = []

    for (a, b, _txt) in intervals:
        sf = int(math.floor((a - t0) / tstep))
        ef = int(math.ceil((b - t0) / tstep))

        # clamp to [0, T]
        sf = max(0, min(sf, T))
        ef = max(0, min(ef, T))
        if ef <= sf:
            continue

        if mode == "start":
            idx = sf
        elif mode == "center":
            idx = (sf + ef) // 2
        else:
            # end event: last frame inside word => ef-1
            idx = ef - 1

        if T > 0:
            idx = max(0, min(idx, T - 1))
        else:
            idx = 0

        ws.append(sf)
        we.append(ef)
        ev.append(idx)

    return np.asarray(ws, np.int64), np.asarray(we, np.int64), np.asarray(ev, np.int64)


# Segment id inference / TextGrid locating
def infer_segment_id(stem: str) -> Optional[str]:
    """
    Accepts stems like:
      - 'segment 1'
      - 'segment_11b'
      - '1'
      - '11b'
    Returns '1'...'11' or '11b' or None.
    """
    s = stem.strip().lower()

    # case 1: direct "1" or "11b"
    if re.fullmatch(r"[0-9]{1,2}b?", s):
        return s

    # case 2: "segment 1" / "segment_11b"
    m = re.search(r"segment[\s_-]*([0-9]{1,2})(b?)", s)
    if m:
        num = m.group(1)
        suf = m.group(2)
        return f"{num}{suf}"

    return None


def find_textgrid(textgrid_dir: Path, seg_id_or_stem: str, verbose: bool = False) -> Optional[Path]:
    """
    Robust finder:
      - if seg_id_or_stem already like 'segment 1' try that
      - else treat it as '1'/'11b' and try 'segment {id}.TextGrid'
      - case-insensitive fallback
    """
    s = seg_id_or_stem.strip()
    candidates = []

    # If user passes "segment 1"
    if s.lower().startswith("segment"):
        candidates.append(textgrid_dir / f"{s}.TextGrid")

        m = re.search(r"segment[\s_-]*([0-9]{1,2}b?)", s.lower())
        if m:
            sid = m.group(1)
            candidates.append(textgrid_dir / f"segment {sid}.TextGrid")
    else:
        candidates.append(textgrid_dir / f"segment {s}.TextGrid")
        candidates.append(textgrid_dir / f"{s}.TextGrid")

    for p in candidates:
        if verbose:
            print("[tg] try:", p)
        if p.exists():
            return p

    # fallback: scan directory
    cand = list(textgrid_dir.glob("*.TextGrid"))
    target1 = f"segment {s}".lower()
    target2 = s.lower()

    for c in cand:
        if c.stem.lower() == target1 or c.stem.lower() == target2:
            return c

    # loose substring fallback
    for c in cand:
        if target2 in c.stem.lower():
            return c

    return None


# Model checkpoint loading
def _strip_prefix(sd: dict, prefix: str) -> dict:
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def extract_state_dict(ckpt) -> dict:
    state = None
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None and "model" in ckpt:
            obj = ckpt["model"]
            if isinstance(obj, dict):
                state = obj
            elif hasattr(obj, "state_dict"):
                state = obj.state_dict()

    if state is None and isinstance(ckpt, dict) and any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt

    if state is None:
        raise RuntimeError(
            f"Cannot extract state_dict from checkpoint. type={type(ckpt)} "
            f"keys={list(ckpt)[:20] if isinstance(ckpt, dict) else 'N/A'}"
        )

    for pref in ("module.", "model.", "net."):
        state = _strip_prefix(state, pref)
    return state


def load_model(args, device: torch.device) -> LSTMFrameSRVSemantic:
    m = LSTMFrameSRVSemantic(
        in_dim=args.in_dim,
        srv_dim=args.srv_dim,
        frame_hidden=args.frame_hidden,
        frame_layers=args.frame_layers,
        dropout=args.dropout,
        word_rep_dim=args.word_rep_dim,
        word_lstm_hidden=args.word_lstm_hidden,
        word_lstm_layers=args.word_lstm_layers,
        rep_dropout=args.rep_dropout,
        word_dropout_p=0.0,
        rep_noise_std=0.0,
    ).to(device)
    m.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = extract_state_dict(ckpt)
    missing, unexpected = m.load_state_dict(state, strict=False)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)
    return m


# Predictor
def compute_mag_change(seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seq: (T,H) or (W,H)
    mag[t] = sum(|h_t|)
    chg[t] = sum(max(|h_t|-|h_{t-1}|,0))
    """
    abs_h = seq.abs()
    mag = abs_h.sum(-1)
    d = abs_h[1:] - abs_h[:-1]
    d = torch.clamp(d, min=0.0)
    chg = d.sum(-1)
    chg = torch.cat([torch.zeros(1, device=chg.device, dtype=chg.dtype), chg], 0)
    return mag, chg


def make_impulse_series(T: int, idx: torch.Tensor, values: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = torch.zeros((T,), device=device, dtype=torch.float32)
    if idx.numel() > 0:
        out[idx] = values.to(torch.float32)
    return out


def make_impulse_matrix(
    T: int,
    idx: torch.Tensor,
    vectors: torch.Tensor,  # (W,H)
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    H = int(vectors.shape[1]) if vectors.numel() else 0
    out = torch.zeros((T, H), device=device, dtype=dtype)
    if idx.numel() > 0 and H > 0:
        out[idx] = vectors.to(dtype)
    return out



def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--stim_dir", type=str, default=r"C:/Dataset/Appleseed/stimuli")
    ap.add_argument("--out_dir", type=str, default=r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors_lstm_sem")
    ap.add_argument("--textgrid_dir", type=str, default=r"C:\linux_project\LENS\appleseed_eelbrain\Appleseed-main\stimuli\text")

    ap.add_argument("--ckpt", type=str, default=r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_SEM_srv_semantic_hier_s000\ckpt\best.pt")
    ap.add_argument("--pattern", type=str, default="*-gammatone100.npz")
    ap.add_argument("--device", type=str, default="cuda")

    # must match training
    ap.add_argument("--in_dim", type=int, default=64)
    ap.add_argument("--srv_dim", type=int, default=2048)
    ap.add_argument("--frame_hidden", type=int, default=512)
    ap.add_argument("--frame_layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--word_rep_dim", type=int, default=256)
    ap.add_argument("--word_lstm_hidden", type=int, default=256)
    ap.add_argument("--word_lstm_layers", type=int, default=1)
    ap.add_argument("--rep_dropout", type=float, default=0.1)

    ap.add_argument("--tail_frames", type=int, default=10)

    ap.add_argument("--event_mode", type=str, default="end", choices=["start", "end", "center"])

    # bool flags (robust)
    ap.add_argument("--save_hidden", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_sem_impulse_hidden", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_debug_words", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    stim_dir = Path(args.stim_dir)
    out_dir = Path(args.out_dir)
    tg_dir = Path(args.textgrid_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = args.device
    if dev.startswith("cuda") and (not torch.cuda.is_available()):
        dev = "cpu"
    device = torch.device(dev)
    print("device:", device)

    model = load_model(args, device)

    npz_files = sorted(stim_dir.glob(args.pattern))
    print("n_files:", len(npz_files))
    if not npz_files:
        raise RuntimeError(f"No files matched: {stim_dir} / {args.pattern}")

    n_total = 0
    n_sem_ok = 0

    for npz_path in npz_files:
        n_total += 1
        stem = npz_path.name.replace("-gammatone100.npz", "")
        data = np.load(npz_path)

        if "x" not in data.files:
            print(f"[skip] {npz_path.name}: missing 'x'")
            continue

        x = data["x"].astype(np.float32, copy=False)  # (T,64)
        t0 = float(data["t0"]) if "t0" in data.files else 0.0
        tstep = float(data["tstep"]) if "tstep" in data.files else 0.01

        T = int(x.shape[0])
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,T,F)
        x_lens = torch.tensor([T], dtype=torch.long, device=device)

        # Layer1
        with torch.inference_mode():
            _, h_BTH, out_lens = model.forward_with_hidden(x_t, x_lens)

        Tb = int(out_lens[0].item())
        h_tH = h_BTH[0, :Tb, :]  # (Tb,Hf)

        frame_mag, frame_chg = compute_mag_change(h_tH)

        frame_out = {
            "t0": np.float64(t0),
            "tstep": np.float64(tstep),
            "env_sr": np.float64(1.0 / tstep),
            "T": np.int64(Tb),
            "frame_h_mag": frame_mag.detach().cpu().numpy().astype(np.float64),
            "frame_h_chg": frame_chg.detach().cpu().numpy().astype(np.float64),
        }

        if args.save_hidden:
            frame_out["frame_h"] = h_tH.detach().cpu().to(torch.float16).numpy()

        frame_path = out_dir / f"{stem}~lstm_frame_predictors.npz"
        np.savez(frame_path, **frame_out)
        if args.verbose:
            print("[save]", frame_path.name, "T=", Tb)

        # Layer2 (event-like)
        seg_id = infer_segment_id(stem)
        if seg_id is None:
            print(f"[warn] {stem}: cannot infer seg_id -> semantic skipped")
            continue

        tg_path = find_textgrid(tg_dir, seg_id, verbose=args.verbose)
        if tg_path is None:
            print(f"[warn] {stem}: TextGrid not found (seg_id={seg_id}) -> semantic skipped")
            continue

        intervals = parse_textgrid_words(tg_path, tier_name="words")
        ws_np, we_np, ev_np = intervals_to_frames(intervals, t0=t0, tstep=tstep, T=Tb, mode=args.event_mode)
        W = int(ws_np.shape[0])

        if W <= 0:
            print(f"[warn] {stem}: no valid word intervals after conversion -> semantic skipped")
            continue

        # sanity (duration roughly match)
        if args.verbose:
            dur_feat = Tb * tstep
            dur_tg = intervals[-1][1] if intervals else 0.0
            print(f"[dbg] {stem}: TG={tg_path.name} dur_feat={dur_feat:.2f}s dur_tg~={dur_tg:.2f}s W={W} event={args.event_mode}")

        word_starts = torch.from_numpy(ws_np[None, :]).long().to(device)# (1,W)
        word_ends = torch.from_numpy(we_np[None, :]).long().to(device) # (1,W)
        word_lens = torch.tensor([W], dtype=torch.long, device=device) # (1,)
        event_idx = torch.from_numpy(ev_np).long().to(device) # (W,)

        # extract word-level sequence c_WH
        with torch.inference_mode():
            reps_h = model._extract_word_reps(
                h=h_BTH,
                out_lens=out_lens,
                word_starts=word_starts,
                word_ends=word_ends,
                word_lens=word_lens,
                tail_frames=int(args.tail_frames),
            )  # (1,W,Hf)

            r = model.word_proj(reps_h) # (1,W,word_rep_dim)

            lens = word_lens.clamp(min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                r, lens.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = model.word_lstm(packed)
            c_BWH, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=W
            )  # (1,W,Hw)

        c_WH = c_BWH[0, :W, :] # (W,Hw)

        word_mag, word_chg = compute_mag_change(c_WH)  # (W,)
        sem_mag_imp = make_impulse_series(Tb, event_idx, word_mag, device)
        sem_chg_imp = make_impulse_series(Tb, event_idx, word_chg, device)

        sem_out = {
            "t0": np.float64(t0),
            "tstep": np.float64(tstep),
            "env_sr": np.float64(1.0 / tstep),
            "T": np.int64(Tb),

            "n_words": np.int64(W),
            "sem_event_mode": np.asarray(args.event_mode, dtype="S"),
            "sem_event_idx": event_idx.detach().cpu().numpy().astype(np.int64),
            "word_starts": ws_np.astype(np.int64),
            "word_ends": we_np.astype(np.int64),

            # event-like predictors
            "sem_c_mag_imp": sem_mag_imp.detach().cpu().numpy().astype(np.float64),
            "sem_c_chg_imp": sem_chg_imp.detach().cpu().numpy().astype(np.float64),
        }

        if args.save_hidden:
            sem_out["sem_c_words"] = c_WH.detach().cpu().to(torch.float16).numpy()

        if args.save_sem_impulse_hidden:
            sem_imp = make_impulse_matrix(Tb, event_idx, c_WH, device=device, dtype=torch.float16)
            sem_out["sem_c_impulse"] = sem_imp.detach().cpu().numpy()

        if args.save_debug_words:
            words = [w for (_, _, w) in intervals]
            xmin_s = np.asarray([a for (a, _, _) in intervals], dtype=np.float64)
            xmax_s = np.asarray([b for (_, b, _) in intervals], dtype=np.float64)
            sem_out["word_text"] = np.asarray(words, dtype=object)
            sem_out["word_xmin_s"] = xmin_s
            sem_out["word_xmax_s"] = xmax_s

        sem_path = out_dir / f"{stem}~lstm_word_predictors.npz"
        np.savez(sem_path, **sem_out)

        n_sem_ok += 1
        if args.verbose:
            print("[save]", sem_path.name, "T=", Tb, f"W={W}")

    print(f"\nDONE. total={n_total} semantic_ok={n_sem_ok} out_dir={out_dir}")


if __name__ == "__main__":
    main()