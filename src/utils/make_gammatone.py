"""
Generate 64-channel gammatone features for LibriSpeech (mirror directory structure)
- 256 -> 64 channels
- resample to 100 Hz (NOT 1000 Hz, which explodes size)
- store float16 to cut size ~50%
"""

from pathlib import Path
import numpy as np

import soundfile as sf
from eelbrain import NDVar, UTS, resample, save, gammatone_bank



LIBRISPEECH_ROOT = Path(r"C:\Dataset\LibriSpeech")
GAMMATONE_ROOT = Path(r"C:\Dataset\LibriSpeech_gammatone64_100Hz")

SUBSETS = ["train-clean-100", "dev-clean", "test-clean"]

# ====== Params ======
FMIN = 20
FMAX = 5000
N_CH = 64
OUT_SR = 100 # 100Hz
DTYPE = np.float16 #减少体积

def load_flac_as_ndvar(path: Path) -> NDVar:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    # 如果是立体声，取平均变单声道
    if x.ndim == 2:
        x = x.mean(axis=1)
    time = UTS(0, 1.0 / sr, len(x))
    return NDVar(x, (time,))

for subset in SUBSETS:
    subset_dir = LIBRISPEECH_ROOT / subset
    if not subset_dir.exists():
        print(f"[Skip] not found: {subset_dir}")
        continue

    # 递归扫所有 flac
    for flac_path in subset_dir.rglob("*.flac"):
        rel = flac_path.relative_to(LIBRISPEECH_ROOT)
        dst = (GAMMATONE_ROOT / rel).with_suffix(".pickle")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            continue

        wav = load_flac_as_ndvar(flac_path)

        # 64-channel gammatone
        gt = gammatone_bank(wav, FMIN, FMAX, N_CH, location="left")

        # 降采样到 100 Hz（原代码 1000 Hz 会超级大）
        gt = resample(gt, OUT_SR)

        # float16 减小体积
        gt = NDVar(gt.x.astype(DTYPE), gt.dims, name=gt.name, info=gt.info)

        save.pickle(gt, dst)

print("Done.")