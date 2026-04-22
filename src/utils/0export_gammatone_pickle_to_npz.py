# export_gammatone_pickle_to_npz.py
from pathlib import Path
import pickle
import numpy as np

STIM_DIR = Path(r"C:/Dataset/Appleseed/stimuli")

for p in STIM_DIR.glob("*-gammatone100.pickle"):
    stem = p.name.replace("-gammatone100.pickle", "")
    nd = pickle.loads(p.read_bytes()) # NDVar (F,T)
    x_ft = nd.x.astype(np.float32, copy=False) # (64,T)
    x_tf = x_ft.T # (T,64)

    out = STIM_DIR / f"{stem}-gammatone100.npz"
    np.savez(out,
             x=x_tf,
             t0=float(nd.time.tmin),
             tstep=float(nd.time.tstep))
    print("[ok]", out.name, x_tf.shape, nd.time.tstep)