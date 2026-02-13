import numpy as np
import eelbrain as eb
from pathlib import Path

PRED_FILE = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors\1~gammatone-8.pickle")

x = eb.load.unpickle(PRED_FILE)   # NDVar
print("file:", PRED_FILE.name)
print("type:", type(x))
print("dims:", x.dims)
print("shape:", x.x.shape)

arr = x.x
print("nonfinite count:", np.sum(~np.isfinite(arr)), "/", arr.size)
print("min/max:", np.nanmin(arr), np.nanmax(arr))

# ---- 找 time 轴（优先按 dim 名称判断）----
time_axis = None
for i, d in enumerate(x.dims):
    if getattr(d, "name", None) == "time":
        time_axis = i
        break
if time_axis is None:
    # 兜底：通常 time 是第一个维度
    time_axis = 0

# ---- 频带轴 = 除 time 外的那个轴（要求是 2D）----
if arr.ndim != 2:
    raise RuntimeError(f"Expected 2D NDVar for gammatone-8, got ndim={arr.ndim}")

feat_axis = 1 - time_axis

# 每个频带在整个时间上的 std
std_per_band = np.nanstd(arr, axis=time_axis)
flat = std_per_band == 0

print("n_bands =", std_per_band.shape[0])
print("flat bands:", np.where(flat)[0].tolist())
print("std_per_band:", std_per_band)

# 如果能拿到频率标签，把 flat bands 对应的频率也打印出来
feat_dim = x.dims[feat_axis]
print("feature dim:", feat_dim)
try:
    vals = feat_dim.values
    print("feature values (first 10):", vals[:10])
    if flat.any():
        print("flat band values:", [vals[i] for i in np.where(flat)[0]])
except Exception as e:
    print("no .values for feature dim:", e)




PRED_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors")
files = sorted(PRED_DIR.glob("*~gammatone-8.pickle"))

print("found", len(files), "files")

bad = []
for f in files:
    x = eb.load.unpickle(f)
    arr = x.x

    # 找 time 轴
    time_axis = None
    for i, d in enumerate(x.dims):
        if getattr(d, "name", None) == "time":
            time_axis = i
            break
    if time_axis is None:
        time_axis = 0

    if arr.ndim != 2:
        bad.append((f.name, "ndim!=2", arr.shape))
        continue

    feat_axis = 1 - time_axis
    n_nonfinite = int(np.sum(~np.isfinite(arr)))
    std_per_band = np.nanstd(arr, axis=time_axis)
    flat_idx = np.where(std_per_band == 0)[0]

    if n_nonfinite > 0 or len(flat_idx) > 0:
        bad.append((f.name, n_nonfinite, flat_idx.tolist(), arr.shape, [getattr(d, "name", None) for d in x.dims]))

print("\n=== BAD SUMMARY ===")
for item in bad[:50]:
    print(item)

print("\nTotal bad:", len(bad), "/", len(files))