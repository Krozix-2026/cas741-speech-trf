# make_cochleagram_tgtc.py
# ------------------------------------------------------------
# Precompute "smart-on-disk" cochleagrams for LibriSpeech:
# - 64 channels
# - env_sr = 100 Hz (controls time resolution / size)
# - float16 storage
# - mirror directory structure
# ------------------------------------------------------------

from pathlib import Path
import sys
import numpy as np
import torch
import torchaudio
import soundfile as sf
import yaml

# ===== paths =====
LIBRISPEECH_ROOT = Path(r"C:\Dataset\LibriSpeech")
OUT_ROOT         = Path(r"C:\Dataset\LibriSpeech_coch64_env100_f16")

TGTC_ROOT = Path(r"C:\linux_project\CAS741\cas741-speech-trf\external\TorchGammatoneCochleagram")
# 让 Python 能 import external 里的 src/
sys.path.insert(0, str(TGTC_ROOT))

from src.gammatone_cochleagram import AudioToCochleagram  # noqa

SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]

# ===== feature params =====
TARGET_SR = 16000     # LibriSpeech audio is 16kHz in general
ENV_SR    = 100       # 100 Hz time grid (size saver)
N_CH      = 64
LOW_LIM   = 20        # Hz (跟你 eelbrain 设定对齐)

DTYPE_SAVE = np.float16

def build_cochleagram_model(device: torch.device):
    # 用仓库自带 YAML 作为“默认配方”，然后改关键参数（最稳：不猜 API）
    yaml_path = TGTC_ROOT / "coch_configs" / "gpu_fir_cochleagram.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    rep = cfg["rep_kwargs"]
    # 覆盖我们关心的核心参数（其余保持作者默认）
    rep["sr"]         = TARGET_SR
    rep["env_sr"]     = ENV_SR
    rep["n_channels"] = N_CH
    rep["low_lim"]    = LOW_LIM
    
    # ==== CRITICAL: disable fixed-length cropping ====
    rep["center_crop"] = False
    
    # 很多配置会用这些字段锁死输入长度（2s -> 200 frames at env_sr=100）
    # 直接删掉，让模型用真实音频长度
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

    print("[Config] center_crop =", rep.get("center_crop"))
    print("[Config] keys containing dur/len/crop/samples:",
          [k for k in rep.keys() if any(s in k.lower() for s in ("dur","len","crop","sample","segment","fixed","max"))])

    coch = AudioToCochleagram(rep)
    coch = coch.to(device)
    coch.eval()
    return coch

def load_flac(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)  # mono
    # (B, C, T)
    wav = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    if sr != TARGET_SR:
        # 用 torchaudio.functional.resample 也可能牵扯依赖；更稳用 scipy
        import scipy.signal as sig
        # 简单高质量重采样：polyphase
        # ratio = TARGET_SR / sr  -> 用 resample_poly
        import math
        g = math.gcd(sr, TARGET_SR)
        up = TARGET_SR // g
        down = sr // g
        y = sig.resample_poly(x, up, down).astype("float32")
        wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)

    return wav

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    coch = build_cochleagram_model(device)

    total = 0
    for subset in SUBSETS:
        subset_dir = LIBRISPEECH_ROOT / subset
        if not subset_dir.exists():
            print(f"[Skip] not found: {subset_dir}")
            continue

        for flac_path in subset_dir.rglob("*.flac"):
            rel = flac_path.relative_to(LIBRISPEECH_ROOT)
            dst = (OUT_ROOT / rel).with_suffix(".npy")
            dst.parent.mkdir(parents=True, exist_ok=True)

            if dst.exists():
                continue

            x = load_flac(flac_path).to(device)

            with torch.inference_mode():
                y = coch(x)  # shape depends on config; usually (B, F, T) or (B, 1, F, T)

            y = y.detach().to("cpu")

            # squeeze batch + possible channel dim
            y = y.squeeze(0)
            if y.ndim == 3 and y.shape[0] == 1:
                y = y.squeeze(0)

            arr = y.to(torch.float16).numpy().astype(DTYPE_SAVE, copy=False)
            np.save(dst, arr)

            total += 1
            if total % 200 == 0:
                print(f"[OK] {total} files done. Last: {rel}")

    print("[Done] All subsets processed.")

if __name__ == "__main__":
    main()