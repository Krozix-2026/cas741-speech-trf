import os
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import soundfile as sf
import inspect

from eelbrain import NDVar, UTS, resample, save
from eelbrain import gammatone_bank


def _read_flac_as_ndvar(flac_path: Path) -> NDVar:
    """Read flac -> eelbrain NDVar (time axis in seconds)."""
    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=False)
    if data.ndim == 2:  # (T, C) -> take first channel
        data = data[:, 0]
    # time axis: tstep = 1/sr, nsamples = len(data)
    wav = NDVar(data, (UTS(0, 1.0 / sr, len(data)),), name="wav")
    return wav


def _maybe_copy_transcript(src_flac: Path, dst_pickle: Path):
    """Copy the chapter transcript file (spk-chap.trans.txt) into dst dir."""
    chap_dir = src_flac.parent
    spk = chap_dir.parent.name
    chap = chap_dir.name
    trans = chap_dir / f"{spk}-{chap}.trans.txt"
    if trans.exists():
        dst_dir = dst_pickle.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_trans = dst_dir / trans.name
        if not dst_trans.exists():
            shutil.copy2(trans, dst_trans)


def convert_one(args):
    """Worker: convert one flac to gammatone pickle."""
    (src_flac, dst_pickle, n_filters, fmin, fmax, out_hz, log1p, copy_trans) = args

    try:
        dst_pickle.parent.mkdir(parents=True, exist_ok=True)
        if dst_pickle.exists():
            return ("skip", str(src_flac))

        wav = _read_flac_as_ndvar(src_flac)

        # gammatone
        gt = gammatone_bank(wav, fmin, fmax, n_filters, location="left")

        # downsample in time (highly recommended for training)
        if out_hz is not None:
            gt = resample(gt, out_hz)

        # optional compression + float32
        if log1p:
            gt.x = np.log1p(np.maximum(gt.x, 0)).astype("float32")
        else:
            gt.x = gt.x.astype("float32")

        save.pickle(gt, dst_pickle)

        if copy_trans:
            _maybe_copy_transcript(src_flac, dst_pickle)

        return ("ok", str(src_flac))

    except Exception as e:
        return ("err", f"{src_flac} -> {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", type=str, default="/media/ramsay/Extreme/LibriSpeech")
    p.add_argument("--out_root", type=str, default="/media/ramsay/Extreme/LibriSpeech_gt")
    p.add_argument("--splits", type=str, nargs="+",
                   default=["train-clean-100", "dev-clean", "test-clean"])
    p.add_argument("--n_filters", type=int, default=64)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=5000.0)
    p.add_argument("--out_hz", type=float, default=100.0,
                   help="Resample gammatone to this Hz (recommend 100 or 200). Use 0 to disable.")
    p.add_argument("--log1p", action="store_true", help="Apply log1p compression to gammatone.")
    p.add_argument("--copy_trans", action="store_true", help="Copy *.trans.txt files into output dirs.")
    p.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1))
    args = p.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_hz = None if args.out_hz == 0 else float(args.out_hz)

    tasks = []
    for split in args.splits:
        split_dir = in_root / split
        if not split_dir.exists():
            print(f"[WARN] split not found: {split_dir}")
            continue

        for src_flac in split_dir.rglob("*.flac"):
            rel = src_flac.relative_to(in_root)# train-clean-100/xxxx/yyy/utt.flac
            dst_pickle = (out_root / rel).with_suffix(".pickle")
            tasks.append((src_flac, dst_pickle, args.n_filters, args.fmin, args.fmax,
                          out_hz, args.log1p, args.copy_trans))

    print(f"[INFO] total flac files: {len(tasks)}")
    print(f"[INFO] writing to: {out_root}")
    print(f"[INFO] n_filters={args.n_filters}, fmin={args.fmin}, fmax={args.fmax}, out_hz={out_hz}, log1p={args.log1p}")
    print(f"[INFO] jobs={args.jobs}")

    ok = skip = err = 0
    with Pool(processes=args.jobs) as pool:
        for status, msg in pool.imap_unordered(convert_one, tasks, chunksize=8):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print("[ERR]", msg)

            if (ok + skip + err) % 500 == 0:
                print(f"[PROG] ok={ok} skip={skip} err={err}")

    print(f"[DONE] ok={ok} skip={skip} err={err}")


if __name__ == "__main__":
    main()
