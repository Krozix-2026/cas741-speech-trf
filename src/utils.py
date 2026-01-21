import warnings
import pickle
import numpy as np


def compress_and_norm_64T(G_64T: np.ndarray, k: float = 10.0) -> np.ndarray:
    """
    Input:  (64, T) non-negative
    Output: (T, 64) float32 ; log-compress + robust per-band [0,1]
    """
    G = np.maximum(G_64T, 0.0)
    Gc = np.log1p(k * G)
    lo = np.quantile(Gc, 0.01, axis=1, keepdims=True)
    hi = np.quantile(Gc, 0.99, axis=1, keepdims=True)
    Gn = (Gc - lo) / (hi - lo + 1e-6)
    Gn = np.clip(Gn, 0.0, 1.0).astype(np.float32)  # (64, T)
    return Gn.T  # -> (T, 64)


def load_pickle_quiet(path: str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        with open(path, "rb") as f:
            return pickle.load(f)

