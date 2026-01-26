import pickle

from pathlib import Path
import re, pickle
import numpy as np
import pandas as pd

CACHE = Path("/media/ramsay/Extreme/Burgundy/eelbrain-cache")

# 你从运行日志里得到的模型编号（举例：baseline=700，RNN=701/702/703）
BASE_ID = 700
RNN_IDS = {
    "rnn_512": 701,
    "rnn_1024": 702,
    "rnn_2048": 703,
}

# 先随便指定一个 score_key（你用第2步确认后再改）
SCORE_KEY = "det"   # 或 "r" 等

def find_cv_files(model_id: int):
    files = []
    for p in CACHE.rglob("*.pickle"):
        s = str(p)
        if f"model{model_id}" in s and "cv.pickle" in s and "50Hz" in s and "superiortemporal" in s:
            files.append(p)
    return sorted(files)

def load_score(p: Path):
    obj = pickle.load(open(p, "rb"))
    # 常见情况：Dataset-like
    if hasattr(obj, "keys") and SCORE_KEY in obj:
        v = obj[SCORE_KEY]
        # v 可能是 Var/NDVar/数组/标量，尽量压成 float
        if hasattr(v, "x"):
            return float(np.mean(v.x))
        try:
            return float(np.mean(v))
        except Exception:
            return float(v)
    # 如果是 dict
    if isinstance(obj, dict) and SCORE_KEY in obj:
        v = obj[SCORE_KEY]
        return float(np.mean(getattr(v, "x", v)))
    # 如果就是数
    if isinstance(obj, (int, float, np.floating)):
        return float(obj)
    raise RuntimeError(f"Unknown cv pickle structure: {p} / {type(obj)}")

def subject_from_path(p: Path):
    m = re.search(r"(R\d{4})", str(p))
    return m.group(1) if m else "UNKNOWN"

# 读 baseline
base_files = find_cv_files(BASE_ID)
base = {subject_from_path(p): load_score(p) for p in base_files}

rows = []
for name, mid in RNN_IDS.items():
    files = find_cv_files(mid)
    rnn = {subject_from_path(p): load_score(p) for p in files}
    common = sorted(set(base) & set(rnn))
    for subj in common:
        rows.append({"model": name, "subject": subj, "base": base[subj], "rnn": rnn[subj], "delta": rnn[subj]-base[subj]})

df = pd.DataFrame(rows)
print(df.groupby("model")["delta"].agg(["mean","std","count"]))
