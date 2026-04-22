# make_appleseed_lstm_predictors.py
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import torch

import torch.nn as nn
from typing import Tuple
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv import LSTMFrameSRV

from eelbrain import NDVar, UTS, Scalar


STIM_DIR = Path(r"C:/Dataset/Appleseed/stimuli")
OUTPUT_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors_lstm")
ENV_SR = 100# 100 Hz
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

CKPT_PATH = Path(r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_baseline_s000\ckpt\last.pt")


IN_DIM = 64
OUT_DIM = 1024 
HIDDEN = 512
LAYERS = 3
DROPOUT = 0.1


def load_model() -> LSTMFrameSRV:
    m = LSTMFrameSRV(
        in_dim=IN_DIM, out_dim=OUT_DIM, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT
    ).to(DEVICE)
    m.eval()

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # print("ckpt:", ckpt)
    print("ckpt['model'] type =", type(ckpt.get("model", None)))
    if isinstance(ckpt.get("model", None), dict):
        print("ckpt['model'] sample keys =", list(ckpt["model"].keys())[:10])

    state = None

    if isinstance(ckpt, dict):

        for k in ("state_dict", "model_state_dict", "net", "weights"):
            # print("k:", k)

            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break

        if state is None and "model" in ckpt:
            obj = ckpt["model"]
            if isinstance(obj, dict):
                state = obj
            elif hasattr(obj, "state_dict"):
                state = obj.state_dict()

    if state is None and isinstance(ckpt, dict) and any(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        state = ckpt

    if state is None:
        raise RuntimeError(
            f"Cannot extract state_dict from checkpoint. "
            f"type={type(ckpt)} keys={list(ckpt)[:20] if isinstance(ckpt, dict) else 'N/A'}"
        )

    def strip_prefix(sd, prefix):
        if all(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    for pref in ("module.", "model.", "net."):
        state = strip_prefix(state, pref)

    missing, unexpected = m.load_state_dict(state, strict=False)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)

    return m


def ndvar_time_like(ref_nd: NDVar, values_1d: np.ndarray, name: str) -> NDVar:
    """Make a 1D NDVar (predictor x time) with same time axis as ref."""
    if values_1d.ndim != 1:
        raise ValueError(values_1d.shape)
    if values_1d.shape[0] != ref_nd.time.nsamples:
        raise ValueError(f"time length mismatch: {values_1d.shape[0]} vs {ref_nd.time.nsamples}")

    feat = Scalar("predictor", [name])
    nd = NDVar(values_1d[np.newaxis, :], (feat, ref_nd.time))
    return nd


def compute_predictors_from_hidden(out_btH: torch.Tensor, T: int) -> tuple[np.ndarray, np.ndarray]:
    """
    out_btH: (1,T,H) torch
    returns:
      magnitude: (T,)
      change:    (T,)
    """
    h = out_btH[0, :T, :]                    # (T,H)
    abs_h = torch.abs(h)                     # (T,H)

    magnitude = abs_h.sum(dim=-1)            # (T,)

    # Δ|h| along time: abs_h[t]-abs_h[t-1]
    d = abs_h[1:] - abs_h[:-1]               # (T-1,H)
    d = torch.clamp(d, min=0.0)              # half-wave rectification
    change = d.sum(dim=-1)                   # (T-1,)

    # 让 change 与时间长度 T 对齐：在 t=0 处补 0
    change = torch.cat([torch.zeros(1, device=change.device), change], dim=0)  # (T,)
    return magnitude.detach().cpu().numpy(), change.detach().cpu().numpy()


def main():
    model = load_model()

    pkls = sorted(STIM_DIR.glob("*-gammatone100.pickle"))
    if not pkls:
        raise RuntimeError(f"No gammatone100 pickle in {STIM_DIR}")

    for p in pkls:
        stem = p.name.replace("-gammatone100.pickle", "")
        out_mag = STIM_DIR / f"{stem}~lstm_magnitude.pickle"
        out_chg = STIM_DIR / f"{stem}~lstm_change.pickle"

        if out_mag.exists() and out_chg.exists():
            print("[skip]", stem)
            continue

        nd: NDVar = pickle.loads(p.read_bytes())
        # nd.x expected (F,T) = (64,T)
        x_ft = nd.x
        if x_ft.shape[0] != IN_DIM:
            raise RuntimeError(f"{p.name}: expected F={IN_DIM}, got {x_ft.shape}")

        # LSTM expects (B,T,F)
        x_tf = x_ft.T.astype(np.float32, copy=False)  # (T,F)
        T = x_tf.shape[0]
        x = torch.from_numpy(x_tf).unsqueeze(0).to(DEVICE)      # (1,T,F)
        x_lens = torch.tensor([T], dtype=torch.long).to(DEVICE)

        with torch.inference_mode():
            _, out, out_lens = model.forward_with_hidden(x, x_lens)

        T_eff = int(out_lens[0].item())
        mag, chg = compute_predictors_from_hidden(out, T_eff)

        # 强制与 nd 的 time 轴一致（T_eff 应该等于 T）
        if T_eff != nd.time.nsamples:
            raise RuntimeError(f"{stem}: T_eff={T_eff} != nd.time.nsamples={nd.time.nsamples}")

        nd_mag = ndvar_time_like(nd, mag.astype(np.float32, copy=False), "lstm_magnitude")
        nd_chg = ndvar_time_like(nd, chg.astype(np.float32, copy=False), "lstm_change")

        out_mag.write_bytes(pickle.dumps(nd_mag, protocol=pickle.HIGHEST_PROTOCOL))
        out_chg.write_bytes(pickle.dumps(nd_chg, protocol=pickle.HIGHEST_PROTOCOL))

        print(f"[ok] {stem}: T={T_eff} saved -> {out_mag.name}, {out_chg.name}")

    print("[done] all predictors created.")


if __name__ == "__main__":
    main()