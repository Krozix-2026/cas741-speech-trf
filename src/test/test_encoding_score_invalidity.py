import numpy as np
import pytest

eelbrain = pytest.importorskip("eelbrain")
from eelbrain import NDVar, UTS, Scalar

from eva.scoring_validation import compute_encoding_score_checked


def test_scoring_rejects_nan_in_predictor_matrix():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 2))

    X[5, 1] = np.nan # inject invalid value

    with pytest.raises(ValueError) as exc_info:
        compute_encoding_score_checked(X, Y)

    msg = str(exc_info.value)
    assert "Scoring error:" in msg
    assert "predictor matrix contains NaN/Inf values" in msg
    assert "1" in msg # conflicting column index


def test_scoring_rejects_constant_response_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 2))

    Y[:, 0] = 7.0 # constant response -> Pearson correlation undefined

    with pytest.raises(ValueError) as exc_info:
        compute_encoding_score_checked(X, Y)

    msg = str(exc_info.value)
    assert "Scoring error:" in msg
    assert "undefined correlation due to constant response signal" in msg
    assert "0" in msg  # bad response column index


def test_scoring_rejects_constant_predictor_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 3))
    Y = rng.normal(size=(20, 2))

    X[:, 2] = -1.0  # constant predictor

    with pytest.raises(ValueError) as exc_info:
        compute_encoding_score_checked(X, Y)

    msg = str(exc_info.value)
    assert "Scoring error:" in msg
    assert "constant predictor" in msg
    assert "2" in msg


def test_scoring_returns_valid_result_for_clean_input():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))

    # make response partially predictable from X
    W = np.array([[0.5, -0.2], [0.1, 0.3], [-0.4, 0.2], [0.2, 0.1]])
    Y = X @ W + 0.05 * rng.normal(size=(50, 2))

    mean_score, scores = compute_encoding_score_checked(X, Y)

    assert np.isfinite(mean_score)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    
    
    
    def test_scoring_rejects_nan_predictor_ndvar():
    time = UTS(0.0, 0.01, 20)

    pred_dim = Scalar("predictor", ["p0", "p1", "p2"])
    resp_dim = Scalar("sensor", ["r0", "r1"])

    X = np.random.randn(3, 20).astype(np.float64)  # (feature, time)
    Y = np.random.randn(2, 20).astype(np.float64)  # (response, time)

    X[1, 7] = np.nan

    pred = NDVar(X, (pred_dim, time))
    resp = NDVar(Y, (resp_dim, time))

    with pytest.raises(ValueError) as exc_info:
        compute_encoding_score_checked(pred, resp)

    msg = str(exc_info.value)
    assert "predictor matrix contains NaN/Inf values" in msg


def test_scoring_rejects_constant_response_ndvar():
    time = UTS(0.0, 0.01, 20)

    pred_dim = Scalar("predictor", ["p0", "p1", "p2"])
    resp_dim = Scalar("sensor", ["r0", "r1"])

    X = np.random.randn(3, 20).astype(np.float64)
    Y = np.random.randn(2, 20).astype(np.float64)

    Y[0, :] = 4.0  # constant response channel

    pred = NDVar(X, (pred_dim, time))
    resp = NDVar(Y, (resp_dim, time))

    with pytest.raises(ValueError) as exc_info:
        compute_encoding_score_checked(pred, resp)

    msg = str(exc_info.value)
    assert "undefined correlation due to constant response signal" in msg
    assert "0" in msg


def test_scoring_rejects_time_grid_mismatch_ndvar():
    pred_time = UTS(0.0, 0.01, 20)
    resp_time = UTS(0.0, 0.02, 20)  # different tstep

    pred_dim = Scalar("predictor", ["p0", "p1"])
    resp_dim = Scalar("sensor", ["r0", "r1"])

    X = np.random.randn(2, 20).astype(np.float64)
    Y = np.random.randn(2, 20).astype(np.float64)

    pred = NDVar(X, (pred_dim, pred_time))
    resp = NDVar(Y, (resp_dim, resp_time))

    with pytest.raises(ValueError, match="not on the same time grid"):
        compute_encoding_score_checked(pred, resp)