# make_appleseed_gammatone64_env100_pickle.py
# ------------------------------------------------------------
# Appleseed 12 wav -> cochleagram (TorchGammatoneCochleagram)
# - 64 channels
# - env_sr = 100 Hz
# - save as eelbrain NDVar pickle: "<stem>-gammatone100.pickle"
#   dims: (frequency, time) where time.tstep = 0.01
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import sys
import pickle
import math

import numpy as np
import torch
import soundfile as sf
import yaml

# ====== user paths ======
IN_DIR = Path(r"C:/Dataset/Appleseed/stimuli")  # contains: 1.wav ... 11.wav 11b.wav
OUT_DIR = Path(r"C:/Dataset/Appleseed/stimuli")  # or your BIDS stimuli dir
# OUT_DIR = Path(r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/stimuli")

TGTC_ROOT = Path(r"C:\linux_project\LENS\TorchGammatoneCochleagram")


sys.path.insert(0, str(TGTC_ROOT))
from src.gammatone_cochleagram import AudioToCochleagram  # noqa

# ===== feature params (match your LibriSpeech training) =====
TARGET_SR = 16000
ENV_SR = 100
N_CH = 64
LOW_LIM = 20  # Hz
DTYPE_SAVE = np.float16

# ===== eelbrain objects for NDVar =====
from eelbrain import NDVar, UTS, Scalar


def erb_space(low_hz: float, high_hz: float, n: int) -> np.ndarray:
    """ERB-rate spaced center frequencies (approx; good for labeling the axis)."""
    # Glasberg & Moore ERB-rate scale
    def hz_to_erb(f):
        return 21.4 * np.log10(4.37e-3 * f + 1.0)

    def erb_to_hz(e):
        return (10 ** (e / 21.4) - 1.0) / 4.37e-3

    lo = hz_to_erb(low_hz)
    hi = hz_to_erb(high_hz)
    erb = np.linspace(lo, hi, n)
    return erb_to_hz(erb)


def build_cochleagram_model(device: torch.device):
    yaml_path = TGTC_ROOT / "coch_configs" / "gpu_fir_cochleagram.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    rep = cfg["rep_kwargs"]
    # override key params to match training
    rep["sr"] = TARGET_SR
    rep["env_sr"] = ENV_SR
    rep["n_channels"] = N_CH
    rep["low_lim"] = LOW_LIM

    # critical: do NOT crop to fixed length
    rep["center_crop"] = False

    # remove any fixed-length/cropping keys if present
    for k in [
        "signal_dur", "signal_duration", "signal_len",
        "segment_dur", "segment_duration", "segment_len",
        "crop_dur", "crop_duration", "crop_len",
        "max_dur", "max_duration", "max_len",
        "n_samples", "num_samples", "audio_len",
        "fixed_len", "fixed_length"
    ]:
        if k in rep:
            print(f"[Config] removing rep_kwargs['{k}']={rep[k]}")
            rep.pop(k)

    print("[Config] sr/env_sr/n_channels/low_lim =",
          rep.get("sr"), rep.get("env_sr"), rep.get("n_channels"), rep.get("low_lim"))
    if "high_lim" in rep:
        print("[Config] high_lim =", rep["high_lim"])

    coch = AudioToCochleagram(rep).to(device)
    coch.eval()
    return coch, rep


def load_wav_mono_16k(path: Path) -> torch.Tensor:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)  # mono

    if sr != TARGET_SR:
        import scipy.signal as sig
        g = math.gcd(sr, TARGET_SR)
        up = TARGET_SR // g
        down = sr // g
        x = sig.resample_poly(x, up, down).astype("float32")

    wav = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    return wav


def nice_sort_key(p: Path):
    # ensure order: 1..11 then 11b
    s = p.stem
    if s.endswith("b") and s[:-1].isdigit():
        return (int(s[:-1]), 1)
    if s.isdigit():
        return (int(s), 0)
    return (10**9, s)


def main():
    assert IN_DIR.exists(), f"IN_DIR not found: {IN_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    coch, rep = build_cochleagram_model(device)

    # infer high_lim for labeling frequency axis (optional)
    high_lim = float(rep.get("high_lim", TARGET_SR / 2))
    freqs = erb_space(LOW_LIM, high_lim, N_CH)
    freq_dim = Scalar("frequency", freqs, "Hz", "%.0f")

    wavs = sorted(IN_DIR.glob("*.wav"), key=nice_sort_key)
    if not wavs:
        raise RuntimeError(f"No .wav files found in {IN_DIR}")

    for wav_path in wavs:
        stem = wav_path.stem  # "1" ... "11b"
        out_path = OUT_DIR / f"{stem}-gammatone{ENV_SR}.pickle"
        if out_path.exists():
            print(f"[Skip] exists: {out_path.name}")
            continue

        print(f"[Run] {wav_path.name} -> {out_path.name}")
        x = load_wav_mono_16k(wav_path).to(device)

        with torch.inference_mode():
            y = coch(x)

        y = y.detach().to("cpu").squeeze(0)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y.squeeze(0)

        # expect (F, T) or (T, F); normalize to (F, T)
        arr = y.numpy()
        if arr.shape[0] != N_CH and arr.shape[1] == N_CH:
            arr = arr.T
        if arr.shape[0] != N_CH:
            raise RuntimeError(f"Unexpected shape for {wav_path.name}: {arr.shape} (expected F={N_CH})")

        arr = arr.astype(DTYPE_SAVE, copy=False)  # reduce disk

        T = arr.shape[1]
        time_dim = UTS(0.0, 1.0 / ENV_SR, T)  # tstep=0.01 for env_sr=100
        nd = NDVar(arr, (freq_dim, time_dim))

        with open(out_path, "wb") as f:
            pickle.dump(nd, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[OK] saved {out_path.name} shape={arr.shape} tstep={time_dim.tstep}")

    print("[Done] All wav processed.")


if __name__ == "__main__":
    main()