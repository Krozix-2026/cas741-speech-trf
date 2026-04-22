# from appleseed import Appleseed

# ROOT = r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new"
# e = Appleseed(ROOT)

# BASE = "gammatone-8"
# EDGE = "gammatone-8 + gammatone-edge30-8"
# PHONE_ANY = EDGE + " + phone-any"
# PHONE_STATS = EDGE + " + phone-surprisal + phone-entropy + phone-phoneme_entropy"

# common = dict(
#     epoch="apple",
#     raw="1-40",
#     inv="fixed-6-MNE-0",
#     samplingrate=50,
#     mask="superiortemporal",
#     tstart=-0.100,
#     tstop=1.000,
#     basis=0.050,
#     error="l2",
#     partitions=4,
#     cv=True,
#     selective_stopping=1,
#     data="source",      # 你要看脑区激活，一般就用 source
#     metric="z",         # 默认 z 更适合统计（也可用 r）
#     pmin="tfce",
#     make=False,
# )

# comparisons = {
#     "edge_gt_base": f"{EDGE} > {BASE}",
#     "phoneany_gt_edge": f"{PHONE_ANY} > {EDGE}",
#     "phonestats_gt_edge": f"{PHONE_STATS} > {EDGE}",
#     # 也可以加：phonestats_gt_phoneany
# }

# # for name, x in comparisons.items():
# #     ds, res = e.load_model_test(x, return_data=True, **common)
# #     print(name, res)

# common_report = dict(common)
# # 这些键你想在调用里单独指定时，就先从 dict 里删掉
# for k in ["pmin", "make", "path_only"]:
#     common_report.pop(k, None)

# for name, x in comparisons.items():
#     print("\n==", name, "==")
#     print("comparison:", x)
#     print(e.show_model_terms(x))
    
    
#     res = e.load_model_test(x, **common_report, make=False)
#     print(res)
#     print("mean Δ:", res.difference.mean())
#     print("max  Δ:", res.difference.max())

#     clusters = res.clusters
#     print(clusters)



import os
os.environ.setdefault("MAYAVI_OFFSCREEN", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import os
from appleseed import Appleseed
from eelbrain import plot, save  # save.pickle

ROOT = r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new"


def plot_phoneme():
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
        mask="superiortemporal",
    )

    BASE = "gammatone-8"
    EDGE = "gammatone-8 + gammatone-edge30-8"
    PHONE_ANY = EDGE + " + phone-any"
    PHONE_STATS = EDGE + " + phone-surprisal + phone-entropy + phone-phoneme_entropy"

    comparisons = {
        "edge_gt_base": f"{EDGE} > {BASE}",
        "phoneany_gt_edge": f"{PHONE_ANY} > {EDGE}",
        "phonestats_gt_edge": f"{PHONE_STATS} > {EDGE}",
    }


    out = Path(ROOT) / "derivatives" / "eelbrain" / "eva" / "mtrf_compare"
    out.mkdir(parents=True, exist_ok=True)

    for name, x in comparisons.items():
        print(name, ":", x)
        res = e.load_model_test(x, **common)   # x 现在是 "A > B"
        print(name, res)

        # 1) 保存结果对象
        save.pickle(res, out / f"{name}.TTestRelated.pickle")

        # 2) 显著性 p-map（p0=0.05 以上都不显示；p1=0.01 更“亮”）
        b = plot.brain.p_map(res, p0=0.05, p1=0.01, surf="inflated", views="lateral")
        b.save_image(out / f"{name}_pmap.png")

        # 3) 只显示显著区域的效应大小（c1-c0）
        diff = res.masked_difference(p=0.05)
        b2 = plot.brain.brain(diff, surf="inflated", views="lateral")
        b2.save_image(out / f"{name}_diff.png")
    
def plot_gpt2_surprisal_multi_roi():
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

    # --- 模型对照 ---
    ONSET = "word_onset"
    ONSET_SURP = "word_onset + gpt2_surp"

    EDGE = "gammatone-8 + gammatone-edge30-8"
    EDGE_ONSET = EDGE + " + word_onset"
    EDGE_ONSET_SURP = EDGE + " + word_onset + gpt2_surp"

    comparisons = {
        "surp_onset_gt_onset": f"{ONSET_SURP} > {ONSET}",
        "surp_edge_onset_gt_edge_onset": f"{EDGE_ONSET_SURP} > {EDGE_ONSET}",
    }

    # --- 你关心的 ROI 列表（先假设驱动，后探索）---
    masks = [
        "superiortemporal",
        "lateraltemporal",
        "ifg",
        "ftp",
        # "aparc",  # 探索性：最后再开
    ]

    out = Path(ROOT) / "derivatives" / "eelbrain" / "eva" / "mtrf_gpt2surp_compare"
    out.mkdir(parents=True, exist_ok=True)

    for mask in masks:
        common_mask = dict(common)
        common_mask["mask"] = mask

        for name, x in comparisons.items():
            tag = f"{name}_{mask}"
            print(tag, ":", x)

            res = e.load_model_test(x, **common_mask)
            save.pickle(res, out / f"{tag}.TTestRelated.pickle")

            b = plot.brain.p_map(res, p0=0.05, p1=0.01, surf="inflated", views="lateral")
            b.save_image(out / f"{tag}_pmap.png")

            diff = res.masked_difference(p=0.05)
            b2 = plot.brain.brain(diff, surf="inflated", views="lateral")
            b2.save_image(out / f"{tag}_diff.png")

if __name__ == "__main__":
    # plot_phoneme()
    plot_gpt2_surprisal_multi_roi()
