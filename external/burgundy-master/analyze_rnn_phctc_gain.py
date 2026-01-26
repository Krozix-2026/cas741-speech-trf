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

# PHCTC_STEM = "Earshot-PhCTC-LibriSpeechGT-64ch-BiLSTM512-L1-ep39-hu-abs"
PHCTC_STEM = "Earshot-PhoneCTC-LibriSpeechGT-64ch-LSTM2048-L1-ep165-hu-abs"
SUM = f"{PHCTC_STEM}-sum"
ON  = f"{PHCTC_STEM}-onset"
FULL = f"{BASE} + {SUM} + {ON}"

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

# 1) 总提升：FULL > BASE
comp_total = f"{FULL} > {BASE}"
tt_total = e.load_model_test(comp_total, make=True, **STG)
summarize_test(tt_total, "PhCTC TOTAL  (full > base)")

# 2) unique contribution: SUM / ONSET
comp_sum = f"{FULL} @ {SUM}"
tt_sum = e.load_model_test(comp_sum, make=True, **STG)
summarize_test(tt_sum, "PhCTC SUM    (unique)")

comp_on = f"{FULL} @ {ON}"
tt_on = e.load_model_test(comp_on, make=True, **STG)
summarize_test(tt_on, "PhCTC ONSET  (unique)")
