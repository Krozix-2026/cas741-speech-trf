import numpy as np
import pytest

eelbrain = pytest.importorskip("eelbrain")
from eelbrain import NDVar, UTS, Scalar


def align_predictor_to_meg_time(meg_time, predictor_ndvar):

    pred_time = predictor_ndvar.time

    if pred_time.nsamples != meg_time.nsamples:
        raise ValueError(
            f"Incompatible time axis: predictor has {pred_time.nsamples} samples, "
            f"but MEG time grid has {meg_time.nsamples} samples"
        )

    if pred_time.tstep != meg_time.tstep:
        raise ValueError(
            f"Incompatible time axis: predictor tstep={pred_time.tstep}, "
            f"but MEG time grid tstep={meg_time.tstep}"
        )

    if pred_time.tmin != meg_time.tmin:
        raise ValueError(
            f"Incompatible time axis: predictor tmin={pred_time.tmin}, "
            f"but MEG time grid tmin={meg_time.tmin}"
        )

    return predictor_ndvar


def test_predictor_alignment_fails_on_time_length_mismatch():
    meg_time = UTS(0.0, 0.01, 100)# 100 samples
    pred_time = UTS(0.0, 0.01, 95)# 95 samples, mismatch

    feat = Scalar("predictor", ["x"])
    predictor = NDVar(
        np.zeros((1, 95), dtype=np.float32),
        (feat, pred_time),
    )

    with pytest.raises(
        ValueError,
        match=r"Incompatible time axis: predictor has 95 samples, but MEG time grid has 100 samples",
    ):
        align_predictor_to_meg_time(meg_time, predictor)


def test_predictor_alignment_fails_on_tstep_mismatch():
    meg_time = UTS(0.0, 0.01, 100)
    pred_time = UTS(0.0, 0.02, 100)# tstep mismatch

    feat = Scalar("predictor", ["x"])
    predictor = NDVar(
        np.zeros((1, 100), dtype=np.float32),
        (feat, pred_time),
    )

    with pytest.raises(
        ValueError,
        match=r"Incompatible time axis: predictor tstep=0.02, but MEG time grid tstep=0.01",
    ):
        align_predictor_to_meg_time(meg_time, predictor)


def test_predictor_alignment_fails_on_tmin_mismatch():
    meg_time = UTS(0.0, 0.01, 100)
    pred_time = UTS(0.1, 0.01, 100)# tmin mismatch

    feat = Scalar("predictor", ["x"])
    predictor = NDVar(
        np.zeros((1, 100), dtype=np.float32),
        (feat, pred_time),
    )

    with pytest.raises(
        ValueError,
        match=r"Incompatible time axis: predictor tmin=0.1, but MEG time grid tmin=0.0",
    ):
        align_predictor_to_meg_time(meg_time, predictor)