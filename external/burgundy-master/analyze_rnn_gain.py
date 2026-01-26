# /media/ramsay/Extreme/burgundy-master/analyze_rnn_gain.py
from burgundy import e

STG = dict(
    raw='ica1-20',
    samplingrate=50,
    cv=True,
    partitions=-4,
    inv='fixed-6-MNE-0',
    group='good2',
    mask='superiortemporal',
    tstart=-0.100,
    tstop=1.000,
    error='l2',
    selective_stopping=1,
)

BASE = "gt-log8 + phone-0v12345"

def full_model(h: int) -> str:
    stem = f"Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs"
    return f"{BASE} + {stem}-sum + {stem}-onset"

def summarize_test(tt, label: str):
    diff = tt.difference
    c1 = tt.c1_mean
    c0 = tt.c0_mean

    diff_m = float(diff.mean())
    c1_m = float(c1.mean())
    c0_m = float(c0.mean())
    rel = diff_m / c0_m if c0_m != 0 else float('nan')

    print(f"\n[{label}]")
    print("baseline mean:", c0_m)
    print("full mean    :", c1_m)
    print("abs uplift   :", diff_m, "(x100 => percentage points if score is fraction)")
    print("rel uplift   :", rel, "(x100 => % relative to baseline)")
    if hasattr(tt, "p") and tt.p is not None:
        print("min p:", float(tt.p.min()))

for h in (512, 1024, 2048):
    full = full_model(h)

    # 1) 总提升：full > base
    comp_total = f"{full} > {BASE}"
    tt_total = e.load_model_test(comp_total, make=False, **STG)
    summarize_test(tt_total, f"{h} TOTAL  (full > base)")

    # 2) unique contribution: sum / onset
    comp_sum = f"{full} @ Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs-sum"
    tt_sum = e.load_model_test(comp_sum, make=False, **STG)
    summarize_test(tt_sum, f"{h} SUM    (unique)")

    comp_on = f"{full} @ Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs-onset"
    tt_on = e.load_model_test(comp_on, make=False, **STG)
    summarize_test(tt_on, f"{h} ONSET  (unique)")
