# jobs_appleseed.py (ASCII-only)
# jobs_appleseed.py
from pathlib import Path
import os
from appleseed import Appleseed

FS_SUBJECTS = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\freesurfer")
os.environ["SUBJECTS_DIR"] = str(FS_SUBJECTS)

e = Appleseed(r"C:\Dataset\Appleseed_BIDS_new")

BASE = dict(
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



# 你已有的 acoustic baseline
M_BASE = "gammatone-8"
M_EDGE = "gammatone-8 + gammatone-edge30-8"

# phone 系列：看你 predictors 里实际有没有这些 key（你 jobs 里之前用过 phone-any 等）
M_PHONE_ANY = f"{M_EDGE} + phone-any"
M_PHONE_STATS = f"{M_EDGE} + phone-surprisal + phone-entropy + phone-phoneme_entropy"

JOBS = [
    e.trf_job(M_BASE, **BASE),
    e.trf_job(M_EDGE, **BASE),
    e.trf_job(M_PHONE_ANY, **BASE),
    e.trf_job(M_PHONE_STATS, **BASE),
]

































# from pathlib import Path
# import os

# FS_SUBJECTS = Path(r"C:\Dataset\Appleseed-BIDS\derivatives\freesurfer")
# os.environ["SUBJECTS_DIR"] = str(FS_SUBJECTS)

# print("[DEBUG] SUBJECTS_DIR =", os.environ["SUBJECTS_DIR"])
# print("[DEBUG] top entries =", [p.name for p in FS_SUBJECTS.iterdir() if p.is_dir()][:10])
# print("[DEBUG] has sub-R2349 dir?", (FS_SUBJECTS / "sub-R2349").exists())

# from appleseed import Appleseed

# e = Appleseed(r"C:\Dataset\Appleseed-BIDS")

# root = Path(e.root)




# # Common settings for quick sanity runs
# BASE = dict(
#     # task="Appleseed",
#     # split="01",
#     epoch="apple",
#     # raw="ica-0-20",
#     raw="1-40",
#     inv="fixed-6-MNE-0",
#     samplingrate=50,
#     cv=True,
#     partitions=-4,
#     tstart=-0.100,
#     tstop=1.000,
#     error="l2",
#     selective_stopping=1,
# )

# STG = {**BASE, "mask": "superiortemporal"}
# FTP = {**BASE, "mask": "ftp"}  # broader lateral set in this pipeline

# GTE8 = "gammatone-8 + gammatone-edge30-8"


# JOBS = [
#     e.trf_job(GTE8, subject="R2349", **STG),
# ]

# JOBS = [
#     e.trf_job(GTE8, **STG),
#     e.trf_job(f"{GTE8} + phone-any", **STG),
#     e.trf_job(f"{GTE8} + phone-surprisal + phone-entropy + phone-phoneme_entropy", **STG),
#     e.trf_job(f"{GTE8} + phone-p0 + phone-p1_", **STG),
#     e.trf_job(f"{GTE8} + phone-p0", **FTP),
# ]


# JOBS = [
#     # 1) Acoustic baseline (checks gammatone predictors)
#     e.trf_job("gte8", **STG),

#     # 2) Acoustic + phone onsets
#     e.trf_job("gte8 + phone-any", **STG),

#     # 3) Acoustic + cohort-style phone stats
#     e.trf_job("gte8 + phone-surprisal + phone-entropy + phone-phoneme_entropy", **STG),

#     # 4) Acoustic + position split
#     e.trf_job("gte8 + phone-p0 + phone-p1_", **STG),

#     # 5) Broader mask smoke test
#     e.trf_job("gte8 + phone-p0", **FTP),
# ]

# JOBS = [
#     e.trf_job("gte8", **{**STG, "split": s})
#     for s in ("01", "02")
# ]

