from pathlib import Path
import pickle

from eelbrain import set_time
from burgundy import e

PRED_DIR = Path(r"C:\Dataset\Burgundy\eelbrain-cache\predictors")
SESSION = "Burgundy"

# 用已验证能跑的 predictor 当模板（选一个就行）
TEMPLATE = "Earshot-sum"

H_LIST = [512, 1024, 2048]
SUF_LIST = ["sum", "onset"]

for subj in e:
    template_path = PRED_DIR / f"{subj} {SESSION}~{TEMPLATE}.pickle"
    template = pickle.load(open(template_path, "rb"))

    for h in H_LIST:
        stem = f"Earshot-LSTM-{h}-OneHot-M1K-train-hu-abs"
        for suf in SUF_LIST:
            code = f"{stem}-{suf}"
            p = PRED_DIR / f"{subj} {SESSION}~{code}-aligned.pickle"

            x = pickle.load(open(p, "rb"))
            x2 = set_time(x, template)  # crop/pad 到模板时间轴

            # sanity check
            if (x2.time.tmin, x2.time.tstep, x2.time.nsamples) != (template.time.tmin, template.time.tstep, template.time.nsamples):
                raise RuntimeError(f"Time mismatch after set_time: {subj} {code}")

            pickle.dump(x2, open(p, "wb"))
            print("aligned:", subj, code, "->", (x2.time.tmin, x2.time.tstep, x2.time.nsamples))
