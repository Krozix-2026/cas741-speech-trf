from pathlib import Path
import numpy as np
from eelbrain import load, save

sub_id = "12"

ROOT = Path(f"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/derivatives/eelbrain/cache/trf/{sub_id}/1-40_emptyroom_fixed-6-MNE-0")

# gpt2_surp + word_onset superiortemporal
# BASE_FILE = ROOT / f"sub-{sub_id}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz superiortemporal word_onset boosting h50 l2 4ptns ss1 cv.pickle"
# FULL_FILE = ROOT / f"sub-{sub_id}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz superiortemporal model4 boosting h50 l2 4ptns ss1 cv.pickle"

# gpt2_surp + word_onset aprac
BASE_FILE = ROOT / f"sub-{sub_id}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz aparc word_onset boosting h50 l2 4ptns ss1 cv.pickle"
FULL_FILE = ROOT / f"sub-{sub_id}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz aparc model4 boosting h50 l2 4ptns ss1 cv.pickle"



base = load.unpickle(BASE_FILE)
full = load.unpickle(FULL_FILE)

# 推荐指标：proportion_explained（更像“解释了多少”）
pe_base = base.proportion_explained
pe_full = full.proportion_explained
dpe = pe_full - pe_base


def to_array(m):
    return np.asarray(m.x) if hasattr(m, "x") else np.asarray(m)

peb = to_array(pe_base)
pef = to_array(pe_full)

mean_base = float(np.nanmean(peb))
mean_full = float(np.nanmean(pef))
improve_A = (mean_full - mean_base) / mean_base




def mean_metric(m):
    if hasattr(m, "x"):
        return float(np.nanmean(np.asarray(m.x)))
    return float(m)

print(f"mean PE baseline: {mean_metric(pe_base):.7f}")
print(f"mean PE full     : {mean_metric(pe_full):.7f}")
print(f"mean PE delta    : {mean_metric(dpe):.7f}")
print(f"improve_A: {improve_A:.3f}")

# 把 delta 存下来，后面画脑图/做group统计更方便
# out = ROOT / "delta_PE_onset_to_onset+gpt2_surp.pickle"
# save.pickle(dpe, out)
# print("saved:", out)