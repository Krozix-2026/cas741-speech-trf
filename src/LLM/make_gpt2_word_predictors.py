# make_gpt2_word_predictors.py
from pathlib import Path
import re
import numpy as np

from eelbrain import NDVar, UTS, save, load



def make_predictors():
    SR = 50  # must match your TRF samplingrate (e.g., 50 Hz)

    SEG_DIR = Path(r"/home/xiaoshao/projects/def-brodbeck/lens/LLM/Appleseed_LLM_alignment/segments")
    BIDS_ROOT = Path(r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new")
    PRED_DIR = BIDS_ROOT / "derivatives" / "predictors"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # This file should contain columns: stimulus, length (seconds)
    # Use the same one your pipeline uses (the one that listed 1..11, 11b).
    STIM_TSV = Path(r"/home/xiaoshao/projects/def-brodbeck/lens/appleseed_eelbrain/Appleseed-main/appleseed/appleseed_stimuli.txt")

    def normalize_seg(seg_raw) -> str:
        # seg_raw can be numpy scalar / python str / bytes
        if isinstance(seg_raw, np.ndarray):
            seg_raw = seg_raw.item()
        if isinstance(seg_raw, bytes):
            seg_raw = seg_raw.decode("utf-8", errors="ignore")
        s = str(seg_raw).strip()

        # common formats: "1", "11b", "segment 1", "segment 11b", "Segment 1"
        s = re.sub(r"^(segment)\s*", "", s, flags=re.IGNORECASE).strip()

        # keep only trailing token like 1 / 11b if still messy
        m = re.search(r"(\d{1,2}b?)$", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        return s

    def read_stim_lengths(tsv_path: Path) -> dict[str, float]:
        tbl = load.tsv(tsv_path, types="fv")  # eelbrain table
        stim = [str(x) for x in tbl["stimulus"]]
        length = [float(x) for x in tbl["length"]]
        return dict(zip(stim, length))

    stim_len = read_stim_lengths(STIM_TSV)
    print("stim_len:", stim_len)

    def zscore_tokens(x: np.ndarray) -> np.ndarray:
        # zscore over tokens (ignoring nan)
        m = np.nanmean(x)
        s = np.nanstd(x)
        if not np.isfinite(s) or s == 0:
            return np.nan_to_num(x, nan=0.0)
        return (x - m) / s

    npz_files = sorted(SEG_DIR.glob("segment *.gpt2_features.npz"))
    # print("npz_files:", npz_files)

    for f in npz_files:
        print("f:", f)
        data = np.load(f, allow_pickle=True)

        seg_raw = data["segment"]
        print("seg_raw:", seg_raw)
        seg = normalize_seg(seg_raw)
        print("seg:", seg)
        
        
        # fallback: parse from filename like "segment 1.gpt2_features"
        if seg not in stim_len:
            m = re.search(r"segment\s*(\d{1,2}b?)", f.stem, flags=re.IGNORECASE)
            if m:
                seg = m.group(1)
        
        # seg might be array scalar; normalize to string like '1' or '11b'
        if isinstance(seg, np.ndarray):
            seg = seg.item()
        seg = str(seg)

        if seg not in stim_len:
            raise RuntimeError(f"Segment {seg} not found in stimuli length table: {STIM_TSV}")

        n = int(round(stim_len[seg] * SR))
        if n <= 0:
            raise RuntimeError(f"Bad length for segment {seg}: {stim_len[seg]} sec")

        t_on = np.asarray(data["t_on"], dtype=float)  # seconds
        surp = np.asarray(data["surprisal"], dtype=float)

        # (optional but recommended) token-wise zscore:
        # surp = zscore_tokens(surp)

        # map to sample indices on 50 Hz grid
        idx = np.rint(t_on * SR).astype(int)  # nearest bin
        keep = (idx >= 0) & (idx < n)
        idx = idx[keep]
        surp = surp[keep]

        # onset impulse
        x_on = np.zeros(n, dtype=float)
        np.add.at(x_on, idx, 1.0)

        # surprisal impulse (NaN -> 0)
        x_s = np.zeros(n, dtype=float)
        surp = np.nan_to_num(surp, nan=0.0)
        np.add.at(x_s, idx, surp)

        time = UTS(0, 1 / SR, n)
        nd_on = NDVar(x_on, time, name="word_onset")
        nd_s  = NDVar(x_s,  time, name="gpt2_surp")

        out_on = PRED_DIR / f"{seg}~word_onset.pickle"
        out_s  = PRED_DIR / f"{seg}~gpt2_surp.pickle"

        save.pickle(nd_on, out_on)
        save.pickle(nd_s, out_s)

        print(f"[OK] {seg}: saved {out_on.name}, {out_s.name} (n={n})")

    print("Done.")


def verify_pre():
    p = r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors\1~gpt2_surp.pickle"
    x = load.unpickle(p)
    print(x, x.time.tmin, x.time.tmax, len(x))
    print("nonzero:", (x.x != 0).sum())
    
    
    import numpy as np

    SR = 50
    npz = r"C:\linux_project\LENS\LLM\Appleseed_LLM_alignment\segments\segment 1.gpt2_features.npz"
    data = np.load(npz, allow_pickle=True)

    t_on = np.asarray(data["t_on"], float)
    surp = np.asarray(data["surprisal"], float)

    idx = np.rint(t_on * SR).astype(int)

    print("n_tokens:", len(t_on))
    print("n_surp_non_nan:", np.isfinite(surp).sum())
    print("n_unique_bins:", len(np.unique(idx)))
    print("n_unique_bins_non_nan:", len(np.unique(idx[np.isfinite(surp)])))


if __name__ == "__main__":
    make_predictors()
    # verify_pre()
