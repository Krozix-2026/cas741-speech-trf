# 2wrap_predictors_npz_to_pickle.py
from pathlib import Path
import pickle
import numpy as np
from eelbrain import NDVar, UTS

STIM_DIR = Path(r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors_lstm_sem")

TARGET = "lstm_word_predictors"
PREDICTOR1 = "sem_c_mag_imp"
PREDICTOR2 = "sem_c_chg_imp"
OUTPUT1 = "lstm2-mag_imp"
OUTPUT2 = "lstm2-chg_imp"

for npz in STIM_DIR.glob(f"*~{TARGET}.npz"):
    stem = npz.name.replace(f"~{TARGET}.npz", "")
    data = np.load(npz)
    mag = np.asarray(data[PREDICTOR1], dtype=np.float64)
    chg = np.asarray(data[PREDICTOR2], dtype=np.float64)
    t0 = float(data["t0"])
    tstep = float(data["tstep"])

    time = UTS(t0, tstep, mag.shape[0])
    nd_mag = NDVar(mag, (time,))
    nd_chg = NDVar(chg, (time,))

    (STIM_DIR / f"{stem}~{OUTPUT1}.pickle").write_bytes(
        pickle.dumps(nd_mag, protocol=pickle.HIGHEST_PROTOCOL)
    )
    (STIM_DIR / f"{stem}~{OUTPUT2}.pickle").write_bytes(
        pickle.dumps(nd_chg, protocol=pickle.HIGHEST_PROTOCOL)
    )
    print("[ok]", stem)