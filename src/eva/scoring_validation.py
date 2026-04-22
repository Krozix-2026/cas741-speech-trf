# evaluation/scoring_validation.py
from __future__ import annotations

import numpy as np


def _same_time_grid(a, b) -> bool:
    if not hasattr(a, "time") or not hasattr(b, "time"):
        return True
    ta, tb = a.time, b.time
    return (
        ta.nsamples == tb.nsamples
        and ta.tstep == tb.tstep
        and ta.tmin == tb.tmin
    )


def _to_2d_time_first(x, *, name: str) -> np.ndarray:
    """
    Convert ndarray or eelbrain NDVar to a 2D matrix with shape (T, D).
    For NDVar, time axis is detected from x.time.nsamples and moved to axis 0.
    Remaining axes are flattened into D.
    """
    if hasattr(x, "x"):  # NDVar-like
        arr = np.asarray(x.x, dtype=np.float64)

        if not hasattr(x, "time"):
            raise ValueError(f"{name} is NDVar-like but has no time axis metadata")

        nsamples = int(x.time.nsamples)
        candidate_axes = [i for i, s in enumerate(arr.shape) if s == nsamples]
        if not candidate_axes:
            raise ValueError(
                f"{name} has no axis matching time.nsamples={nsamples}; shape={arr.shape}"
            )

        # Prefer the last matching axis as time, common for Eelbrain data
        time_axis = candidate_axes[-1]
        arr = np.moveaxis(arr, time_axis, 0)  # (T, ...)

        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(arr.shape[0], -1)  # flatten non-time dims

        return arr

    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got shape={arr.shape}")
    return arr


def compute_encoding_score_checked(predictor, response):
    """
    Simple encoding-score computation with explicit invalidity checks.

    Input:
        predictor: ndarray/NDVar, interpreted as (T, P) after conversion
        response:  ndarray/NDVar, interpreted as (T, R) after conversion

    Output:
        mean_score: float
        per_response_scores: np.ndarray with shape (R,)

    Error cases:
        - predictor contains NaN/Inf
        - response contains NaN/Inf
        - predictor/response time grids mismatch
        - predictor/response number of time samples mismatch
        - any predictor column is constant
        - any response column is constant
        - computed correlation is NaN/Inf
    """
    if not _same_time_grid(predictor, response):
        raise ValueError("Scoring error: predictor and response are not on the same time grid")

    X = _to_2d_time_first(predictor, name="predictor")
    Y = _to_2d_time_first(response, name="response")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Scoring error: time dimension mismatch between predictor and response "
            f"({X.shape[0]} vs {Y.shape[0]})"
        )

    bad_pred_cols = np.unique(np.where(~np.isfinite(X))[1])
    if bad_pred_cols.size > 0:
        raise ValueError(
            f"Scoring error: predictor matrix contains NaN/Inf values in column(s) "
            f"{bad_pred_cols.tolist()}"
        )

    bad_resp_cols = np.unique(np.where(~np.isfinite(Y))[1])
    if bad_resp_cols.size > 0:
        raise ValueError(
            f"Scoring error: response matrix contains NaN/Inf values in column(s) "
            f"{bad_resp_cols.tolist()}"
        )

    const_pred_cols = np.where(np.nanstd(X, axis=0) == 0)[0]
    if const_pred_cols.size > 0:
        raise ValueError(
            f"Scoring error: undefined regression/correlation due to constant predictor "
            f"signal in column(s) {const_pred_cols.tolist()}"
        )

    const_resp_cols = np.where(np.nanstd(Y, axis=0) == 0)[0]
    if const_resp_cols.size > 0:
        raise ValueError(
            f"Scoring error: undefined correlation due to constant response signal in "
            f"column(s) {const_resp_cols.tolist()}"
        )

    # Ordinary least squares encoding model
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ beta

    scores = []
    for j in range(Y.shape[1]):
        r = np.corrcoef(Y_hat[:, j], Y[:, j])[0, 1]
        if not np.isfinite(r):
            raise ValueError(
                f"Scoring error: computed correlation is invalid for response column {j}"
            )
        scores.append(r)

    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean()), scores