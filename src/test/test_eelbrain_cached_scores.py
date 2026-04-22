from pathlib import Path
import os
import numpy as np
import pytest

eelbrain = pytest.importorskip("eelbrain")
from eelbrain import load


SUB_ID = os.getenv("APPLESEED_SUB_ID", "12")
ROOT = Path(
    os.getenv(
        "APPLESEED_TRF_CACHE_ROOT",
        f"../../../Appleseed_BIDS_new/derivatives/eelbrain/cache/trf/{SUB_ID}/1-40_emptyroom_fixed-6-MNE-0",
    )
)

BASE_FILE = ROOT / (
    f"sub-{SUB_ID}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz "
    f"superiortemporal model0 boosting h50 l2 4ptns ss1 cv.pickle"
)
FULL_FILE = ROOT / (
    f"sub-{SUB_ID}_ses-emptyroom_run-01_meg nobl -100-1000 50Hz "
    f"superiortemporal model3 boosting h50 l2 4ptns ss1 cv.pickle"
)


def _to_array(m):
    return np.asarray(m.x) if hasattr(m, "x") else np.asarray(m)


@pytest.mark.integration
@pytest.mark.skipif(not BASE_FILE.exists(), reason="BASE_FILE not found")
@pytest.mark.skipif(not FULL_FILE.exists(), reason="FULL_FILE not found")
def test_cached_proportion_explained_contains_no_nan_or_inf():
    base = load.unpickle(BASE_FILE)
    full = load.unpickle(FULL_FILE)

    peb = _to_array(base.proportion_explained)
    pef = _to_array(full.proportion_explained)

    assert np.isfinite(peb).all(), "Invalid baseline score: proportion_explained contains NaN/Inf"
    assert np.isfinite(pef).all(), "Invalid full-model score: proportion_explained contains NaN/Inf"