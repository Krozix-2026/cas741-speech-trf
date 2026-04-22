# jobs_appleseed.py
from pathlib import Path
import os
from appleseed import Appleseed

FS_SUBJECTS = Path(r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new/derivatives/freesurfer")
os.environ["SUBJECTS_DIR"] = str(FS_SUBJECTS)

e = Appleseed(r"/home/xiaoshao/projects/def-brodbeck/datasets/Appleseed_BIDS_new")

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
    # mask="superiortemporal",
)

# lstm
MASKS = {
    "stg": "superiortemporal", 
    "wb": "aparc",              # whole-brain parcellation
}

MODEL_KEYS = [
    # acoustic baseline
    "gte8",

    # LSTM vs acoustic
    "gte8_lstm_mag",
    "gte8_lstm_chg",
    "gte8_lstm_both",

    # lexical boundary control
    "gte8_word_onset",
    "gte8_word_onset_lstm_both",
]


JOBS = []
for tag, mask in MASKS.items():
    for mk in MODEL_KEYS:
        job = e.trf_job(mk, mask=mask, **BASE)
        JOBS.append(job)


# GPT-2
# MASKS = {
#     "stg": "superiortemporal", 
#     "wb": "aparc",              # whole-brain parcellation
# }


# MODEL_KEYS = [
#     # language-only sanity
#     "word_onset",
#     "word_onset_surp",

#     # recommended: acoustic-controlled contrast
#     "gte8_word_onset",
#     "gte8_word_onset_surp",
# ]

# JOBS = []
# for tag, mask in MASKS.items():
#     for mk in MODEL_KEYS:
#         JOBS.append(e.trf_job(mk, mask=mask, **BASE))



# # acoustic baseline
# M_BASE = "gammatone-8"
# M_EDGE = "gammatone-8 + gammatone-edge30-8"

# # phone 系列：看你 predictors 里实际有没有这些 key（你 jobs 里之前用过 phone-any 等）
# M_PHONE_ANY = f"{M_EDGE} + phone-any"
# M_PHONE_STATS = f"{M_EDGE} + phone-surprisal + phone-entropy + phone-phoneme_entropy"

# models = {
#     "gte8": "gammatone-8",
#     "gte8_edge": "gammatone-8 + gammatone-edge30-8",
#     "gte8_edge_phone_any": "gammatone-8 + gammatone-edge30-8 + phone-any",
#     "gte8_edge_phone_stats": "gammatone-8 + gammatone-edge30-8 + phone-surprisal + phone-entropy + phone-phoneme_entropy",
# }



# JOBS = [
#     e.trf_job("gte8", **BASE),
#     e.trf_job("gte8_edge", **BASE),
#     e.trf_job("gte8_edge_phone_any", **BASE),
#     e.trf_job("gte8_edge_phone_stats", **BASE),
# ]





















