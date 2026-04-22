# plot_on_nibi_gpt2_multi_roi.py
from __future__ import annotations
import os
from pathlib import Path
import time

# headless rendering (no X needed)
os.environ["MAYAVI_OFFSCREEN"] = "1"
os.environ["ETS_TOOLKIT"] = "qt"
os.environ["QT_API"] = "pyqt5"

from appleseed import Appleseed
from eelbrain import plot, save


ROOT = r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new"
SUBJECTS_DIR = "/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/derivatives/freesurfer"

e = Appleseed(ROOT)

common = dict(
    epoch="apple",
    raw="1-40",
    inv="fixed-6-MNE-0",
    samplingrate=50,
    cv=True,
    partitions=4,
    tstart=-0.100,
    tstop=1.000,
    error="l2",
    selective_stopping=1,
)

# -------------------------
# Model comparisons
# -------------------------
# 语言-only：onset vs onset+surprisal
ONSET = "word_onset"
ONSET_SURP = "word_onset + gpt2_surp"

# 更推荐：声学控制后 surprisal 增益（写论文更硬）
EDGE = "gammatone-8 + gammatone-edge30-8"
EDGE_ONSET = EDGE + " + word_onset"
EDGE_ONSET_SURP = EDGE + " + word_onset + gpt2_surp"

comparisons = {
    # surprisal 在 word onset 基线上增加的提升
    "gpt2surp_onset_gt_onset": f"{ONSET_SURP} > {ONSET}",

    # 在声学 + onset 控制后，surprisal 的额外提升
    "gpt2surp_edge_onset_gt_edge_onset": f"{EDGE_ONSET_SURP} > {EDGE_ONSET}",
}

# -------------------------
# ROI masks (batch)
# -------------------------
# 先假设驱动 ROI（推荐）
MASKS = [
    "superiortemporal",
    "lateraltemporal",
    "ifg",
    "ftp",
]

# 探索性全脑：很重，建议最后单独开
# MASKS.append("aparc")

out = Path(ROOT) / "derivatives" / "eelbrain" / "eva" / "mtrf_gpt2surp_multiROI_figs"
out.mkdir(parents=True, exist_ok=True)


def render(res, tag: str):
    """Render p-map and masked difference for both hemispheres."""
    for hemi in ("lh", "rh"):
        b = plot.brain.p_map(
            res,
            p0=0.05,
            p1=0.01,
            surf="inflated",
            views="lateral",
            hemi=hemi,
            subjects_dir=SUBJECTS_DIR,
        )
        b.set_size(1600, 1000)
        time.sleep(0.2)
        b.save_image(out / f"{tag}_pmap_{hemi}.png", mode="rgb", antialiased=True)
        b.close()

        md = res.masked_difference(p=0.05)
        b2 = plot.brain.brain(
            md,
            surf="inflated",
            views="lateral",
            hemi=hemi,
            subjects_dir=SUBJECTS_DIR,
        )
        b2.set_size(1600, 1000)
        time.sleep(0.2)
        b2.save_image(out / f"{tag}_diff_{hemi}.png", mode="rgb", antialiased=True)
        b2.close()


# （可选）快速 sanity：确认模型表达式能被解析（不等于一定命中旧 cache，但至少不会“unknown term”）
print("=== Sanity: model terms ===")
try:
    e.show_model_terms(ONSET)
    e.show_model_terms(ONSET_SURP)
    e.show_model_terms(EDGE_ONSET)
    e.show_model_terms(EDGE_ONSET_SURP)
except Exception as ex:
    print("[WARN] show_model_terms failed:", ex)
print("===========================")


for mask in MASKS:
    common_mask = dict(common)
    common_mask["mask"] = mask

    for name, x in comparisons.items():
        tag = f"{name}__{mask}"
        print("\n==", tag, "==")
        print("expr:", x)

        try:
            res = e.load_model_test(x, **common_mask)  # returns TTestRelated
        except Exception as ex:
            print("[ERROR] load_model_test failed for", tag, "->", ex)
            continue

        print(res)
        save.pickle(res, out / f"{tag}.TTestRelated.pickle")
        render(res, tag)

print("[OK] wrote to", out)