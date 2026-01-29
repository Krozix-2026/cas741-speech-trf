# jobs_appleseed.py
# Minimal smoke-test jobs for Appleseed TRF pipeline


from appleseed_eelbrain.pipeline.appleseed_pipeline import e


# ========= Quick configs =========
# 先用 no-ICA 的 raw（你现在 pipeline 里跑通的那个）；等 ICA/tSSS 都准备好再改成 'ica1-20'
BASE = {
    "raw": "noica1-20",          # <- 按你 pipeline 里真实存在的 raw key 改
    "samplingrate": 50,          # 先低一点更快
    "cv": True,
    "partitions": -4,
    "inv": "fixed-1-MNE-0",      # 你 pipeline defaults 里用的 inv
    "tstart": -0.100,
    "tstop": 1.000,
    "error": "l2",
    "selective_stopping": 1,
    # "epoch": "cont",           # 不写也行（defaults 已经是 cont）
}

# 用你 parcs 里真实存在的 mask 名字
WHOLEBRAIN = {**BASE, "mask": "wholebrain-2"}
STG = {**BASE, "mask": "superiortemporal"}



JOBS = [
    # 1) 最基础：声学 baseline（确保 gammatone predictors 能被读到）
    e.trf_job("gt-log8-edge30", **STG),

    # 2) 声学 + phone onset（列名 phone-any 在 phone.pickle 里存在）
    e.trf_job("gt-log8-edge30 + phone-any", **STG),

    # 3) 声学 + phone cohort（你 phone.pickle 里有 surprisal/entropy/phoneme_entropy）
    e.trf_job("gt-log8-edge30 + phone-surprisal + phone-entropy + phone-phoneme_entropy", **STG),

    # 4) 声学 + phone position split（phone-p0 / phone-p1_）
    e.trf_job("gt-log8-edge30 + phone-p0 + phone-p1_", **STG),

    # 5) （可选）Wholebrain 跑一个，确认 mask/forward/inv 都正常
    e.trf_job("gt-log8-edge30 + phone-p0", **WHOLEBRAIN),
]