from pathlib import Path
from eelbrain import load

ROOT = Path("/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/derivatives/eelbrain/cache/trf/01/1-40_emptyroom_fixed-6-MNE-0")

# 只看名字里包含 model 的（你也可以换成 "cv.pickle" 或其他关键词）
files = sorted([p for p in ROOT.glob("*.pickle") if "model" in p.name.lower()])

print("n_files =", len(files))
for p in files[:10]:
    print(" -", p.name)

if not files:
    raise SystemExit("No model*.pickle found")

p = files[0]
x = load.unpickle(p)

print("\n=== OPEN ===")
print("file:", p.name)
print("type:", type(x))

# 通用“探针”：看看有哪些字段/键
if isinstance(x, dict):
    print("dict keys:", list(x.keys())[:50])
elif hasattr(x, "keys"):
    try:
        print("keys():", list(x.keys())[:50])
    except Exception as e:
        print("keys() failed:", e)

# 看看对象属性（避免刷屏，只显示前 50 个）
attrs = [a for a in dir(x) if not a.startswith("_")]
print("attrs:", attrs[:50])

# 如果是 eelbrain Dataset，常用信息
if hasattr(x, "n_cases"):
    print("n_cases:", getattr(x, "n_cases"))
if hasattr(x, "info"):
    print("info keys:", list(getattr(x, "info").keys())[:50])