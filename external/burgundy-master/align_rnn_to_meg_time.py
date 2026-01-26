# align_rnn_to_meg_time.py
from pathlib import Path
import pickle

from eelbrain import set_time, concatenate
from burgundy import e

PRED_DIR = Path("C:\Dataset\Burgundy\eelbrain-cache\predictors")
SESSION = "Burgundy"

# H_LIST = [512, 1024, 2048]
# SUF_LIST = ["sum", "onset"]

# def get_y_template(subj: str):
#     # 这里 samplingrate=100 对应你的 predictor tstep=0.01
#     ds = e.load_epochs(subjects=subj, epoch="cont", samplingrate=100, data="sensor", ndvar=True)
#     y = ds["meg"]  # 如果你这列名不是 meg，就改成实际那列
#     if y.__class__.__name__ == "Datalist":
#         # 多段的话先拼起来（与你之前调试方式一致）
#         y_all = concatenate(list(y), dim="time", tmin="first")
#     else:
#         y_all = y
#     return y_all  # NDVar（带 time 轴）

# for subj in e:
#     y_template = get_y_template(subj)

#     for h in H_LIST:
#         stem = f"Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs"
#         for suf in SUF_LIST:
#             code = f"{stem}-{suf}"
#             p = PRED_DIR / f"{subj} {SESSION}~{code}.pickle"

#             x = pickle.load(open(p, "rb"))

#             # 对齐（crop/pad 到 y_template 的 time 轴）
#             x2 = set_time(x, y_template)

#             # 可选：先备份一次
#             # p_bak = p.with_suffix(p.suffix + ".bak")
#             # if not p_bak.exists():
#             #     pickle.dump(x, open(p_bak, "wb"))

#             pickle.dump(x2, open(p, "wb"))

#             print("aligned", subj, code, "->",
#                   (x2.time.tmin, x2.time.tstep, x2.time.nsamples))

import time

subj, sess = "R2647", "Burgundy"
code = "Earshot-LSTM-1024-OneHot-M1K-train-hu-abs-onset"

p = PRED_DIR / f"{subj} {sess}~{code}.pickle"
st = p.stat()

# print("e.root =", e.root)
print("path =", p)
print("mtime =", time.ctime(st.st_mtime), "size =", st.st_size)

x = pickle.load(open(p, "rb"))
print("X time =", x.time)
