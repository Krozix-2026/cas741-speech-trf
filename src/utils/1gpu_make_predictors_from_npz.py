# 1gpu_make_predictors_from_npz.py
from pathlib import Path
import numpy as np
import torch

import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from network.lstm_frame_srv import LSTMFrameSRV


STIM_DIR = Path(r"C:/Dataset/Appleseed/stimuli")
OUTPUT_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors_lstm")
ENV_SR = 100 # 100 Hz
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

model = load_model()


def compute_mag_change(out_tH: torch.Tensor):
    abs_h = out_tH.abs() # (T,H)
    mag = abs_h.sum(-1) # (T,)
    d = abs_h[1:] - abs_h[:-1] # (T-1,H)
    d = torch.clamp(d, min=0.0) # half-wave
    chg = d.sum(-1) # (T-1,)
    chg = torch.cat([torch.zeros(1, device=chg.device), chg], 0) # (T,)
    return mag, chg

for npz in STIM_DIR.glob("*-gammatone100.npz"):
    stem = npz.name.replace("-gammatone100.npz", "")
    data = np.load(npz)
    x = data["x"].astype(np.float32, copy=False) # (T,64)
    print("x:", x.shape)
    t0 = float(data["t0"])
    tstep = float(data["tstep"])

    T = x.shape[0]
    x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE) # (1,T,64)
    x_lens = torch.tensor([T], dtype=torch.long).to(DEVICE)

    with torch.inference_mode():
        _, out, out_lens = model.forward_with_hidden(x_t, x_lens)
    out_tH = out[0, :out_lens[0].item(), :] # (T,H)

    mag, chg = compute_mag_change(out_tH)
    mag = mag.detach().cpu().numpy().astype(np.float64)
    chg = chg.detach().cpu().numpy().astype(np.float64)

    out_path = OUTPUT_DIR / f"{stem}-lstm_predictors.npz"
    np.savez(out_path, magnitude=mag, change=chg, t0=t0, tstep=tstep)
    print("[ok]", out_path.name, mag.shape, chg.shape, "device", DEVICE)