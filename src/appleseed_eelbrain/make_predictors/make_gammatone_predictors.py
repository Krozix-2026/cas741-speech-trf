from pathlib import Path
import os
import re
import numpy as np
import eelbrain as eb


DATA_ROOT = Path(os.environ.get("ALICE_ROOT", r"C:\Dataset\Appleseed")).expanduser()
STIMULUS_DIR = DATA_ROOT / "stimuli"
PREDICTOR_DIR = Path("C:\Dataset\Appleseed_BIDS_new\derivatives\predictors")
# PREDICTOR_DIR.mkdir(exist_ok=True)

pickle_paths = sorted(STIMULUS_DIR.glob("*-gammatone1000.pickle"))
if not pickle_paths:
    raise FileNotFoundError(f"No '*-gammatone1000.pickle' found in {STIMULUS_DIR}")

def natural_key(p: Path):
    stem = p.name.replace("-gammatone1000.pickle", "")
    parts = re.split(r"(\d+)", stem)
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key

pickle_paths = sorted(pickle_paths, key=natural_key)

def sanitize_ndvar(gt, key, mode="clip"):
    """Ensure gt.x is finite and non-negative.
    mode='clip': negative -> 0 (适合本来就应是能量谱的情况)
    mode='abs' : negative -> abs (如果你发现 gt 像滤波波形而不是能量)
    """
    gt = gt.copy()
    x = np.asarray(gt.x, dtype=float)

    # 1) NaN/inf -> 0
    bad = ~np.isfinite(x)
    if bad.any():
        print(f"[WARN] {key}: nonfinite in raw gammatone = {bad.sum()} -> set to 0")
        x = x.copy()
        x[bad] = 0.0

    # 2) negative handling
    neg = x < 0
    if neg.any():
        frac = neg.mean()
        print(f"[WARN] {key}: negative values in gammatone = {frac*100:.3f}% (min={x.min():.6g})")
        x = x.copy()
        if mode == "abs":
            x = np.abs(x)
        else:
            x[neg] = 0.0

    gt.x = x
    return gt

def clean_ndvar(x: eb.NDVar, *, nonneg: bool, key: str):
    arr = np.array(x.x, dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if nonneg:
        arr[arr < 0] = 0.0
    return eb.NDVar(arr, x.dims, name=x.name)

def clean_edge(x: eb.NDVar, key: str):
    arr = np.array(x.x, dtype=np.float64, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.abs(arr)  # 全波整流，保留“变化强度”
    return eb.NDVar(arr, x.dims, name=x.name)


for p in pickle_paths:
    key = p.name.replace("-gammatone1000.pickle", "")
    gt = eb.load.unpickle(p)

    # 0) 先把原始 gammatone1000 变成 100% finite；是否强制非负取决于它本质
    #    一般 gammatone“能量谱”应非负；如果你发现负值很多，说明它可能是滤波输出，那就先 abs 再 log
    gt = clean_ndvar(gt, nonneg=False, key=f"{key} raw")

    # 如果 raw 里负值比例很大（>1%），我建议打开这一行，把它变成“幅值谱”再 log
    # （避免后面 log/幂运算遇到负数）
    # gt = eb.NDVar(np.abs(gt.x), gt.dims, name=gt.name)

    # 1) log 特征：一定要在 log 前保证输入 >= 0
    #    safest: 用 abs/clip 之后再 log1p
    gt_nonneg = clean_ndvar(gt, nonneg=True, key=f"{key} nonneg")   # 负值 -> 0
    gt_log = (gt_nonneg + 1).log()
    gt_log = clean_ndvar(gt_log, nonneg=True, key=f"{key} log")     # log 后再清一次（必须）

    # 2) edge/onset 特征：edge_detector 之后用 abs（别 clip，否则更容易出现大片 0 -> flat）
    gt_edge = eb.edge_detector(gt_log, c=30)
    gt_edge = clean_edge(gt_edge, key=f"{key} edge")                # abs + finite

    # 3) 1-band（envelope）
    g1 = gt_log.sum("frequency")
    g1 = clean_ndvar(g1, nonneg=True, key=f"{key} g1")
    eb.save.pickle(g1, PREDICTOR_DIR / f"{key}~gammatone-1.pickle")

    e1 = gt_edge.sum("frequency")
    e1 = clean_ndvar(e1, nonneg=True, key=f"{key} e1")
    eb.save.pickle(e1, PREDICTOR_DIR / f"{key}~gammatone-edge30-1.pickle")

    # 4) 8-band（你 boosting 用的核心）
    g8 = gt_log.bin(nbins=8, func="sum", dim="frequency")
    g8 = clean_ndvar(g8, nonneg=True, key=f"{key} g8")              # bin 后再清（必须）
    eb.save.pickle(g8, PREDICTOR_DIR / f"{key}~gammatone-8.pickle")

    e8 = gt_edge.bin(nbins=8, func="sum", dim="frequency")
    e8 = clean_ndvar(e8, nonneg=True, key=f"{key} e8")              # abs 后本来非负，但清一下无害
    eb.save.pickle(e8, PREDICTOR_DIR / f"{key}~gammatone-edge30-8.pickle")

    # 5) 可选：linear 8-bin（用 raw 的非负版，避免负数）
    glin8 = gt_nonneg.bin(nbins=8, func="sum", dim="frequency")
    glin8 = clean_ndvar(glin8, nonneg=True, key=f"{key} glin8")
    eb.save.pickle(glin8, PREDICTOR_DIR / f"{key}~gammatone-lin-8.pickle")

    # 6) 可选：powerlaw 8-bin（一定要非负，否则 **0.6 会 NaN）
    gt_pow = eb.NDVar(gt_nonneg.x ** 0.6, gt_nonneg.dims, name=gt_nonneg.name)
    gt_pow = clean_ndvar(gt_pow, nonneg=True, key=f"{key} pow")
    gpow8 = gt_pow.bin(nbins=8, func="sum", dim="frequency")
    gpow8 = clean_ndvar(gpow8, nonneg=True, key=f"{key} gpow8")
    eb.save.pickle(gpow8, PREDICTOR_DIR / f"{key}~gammatone-pow-8.pickle")

    print(f"Done: {key}")
