import os
import pickle
from pathlib import Path

import numpy as np
import pytest


PREDICTOR_ROOT = Path(
    os.getenv(
        "APPLESEED_PREDICTOR_ROOT",
        r"C:\Dataset\Appleseed_BIDS_new\derivatives\predictors",
    )
)


EXPECTED_SUBJECT1_PREDICTORS = {
    "1~c5phone.pickle": 2,
    "1~gammatone-8.pickle": 2,
}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_array_and_time_info(obj):
    """
        arr: np.ndarray
        time_len: int
        tstep: Optional[float]
        tmin: Optional[float]
    """
    # NDVar-like object
    if hasattr(obj, "x") and hasattr(obj, "time"):
        arr = np.asarray(obj.x)
        time = obj.time
        time_len = int(time.nsamples)
        tstep = getattr(time, "tstep", None)
        tmin = getattr(time, "tmin", None)
        return arr, time_len, tstep, tmin

    # Plain array-like object
    arr = np.asarray(obj)
    if arr.ndim == 0:
        raise ValueError("Predictor object is scalar, expected array-like predictor.")

    time_len = int(arr.shape[-1])
    return arr, time_len, None, None


def test_predictor_root_exists():
    assert PREDICTOR_ROOT.exists(), f"Predictor root does not exist: {PREDICTOR_ROOT}"
    assert PREDICTOR_ROOT.is_dir(), f"Predictor root is not a directory: {PREDICTOR_ROOT}"


def test_predictor_pickle_files_exist():
    pkls = sorted(PREDICTOR_ROOT.glob("*.pickle"))
    assert len(pkls) > 0, f"No .pickle predictor files found in {PREDICTOR_ROOT}"


def test_predictor_file_suffixes_are_pickle():
    pkls = sorted(PREDICTOR_ROOT.glob("*.pickle"))
    assert len(pkls) > 0, f"No .pickle predictor files found in {PREDICTOR_ROOT}"

    for p in pkls:
        assert p.suffix == ".pickle", f"Unexpected suffix for predictor file: {p}"


@pytest.mark.parametrize("filename, expected_ndim", EXPECTED_SUBJECT1_PREDICTORS.items())
def test_subject1_required_predictors_exist(filename, expected_ndim):
    path = PREDICTOR_ROOT / filename
    assert path.exists(), f"Required predictor file is missing: {path}"
    assert path.is_file(), f"Predictor path is not a file: {path}"
    assert path.suffix == ".pickle", f"Predictor file must end with .pickle: {path}"


@pytest.mark.parametrize("filename, expected_ndim", EXPECTED_SUBJECT1_PREDICTORS.items())
def test_subject1_predictors_are_loadable_and_have_expected_ndim(filename, expected_ndim):
    path = PREDICTOR_ROOT / filename
    obj = _load_pickle(path)

    arr, time_len, _, _ = _extract_array_and_time_info(obj)

    assert arr.ndim == expected_ndim, (
        f"{filename} has ndim={arr.ndim}, expected {expected_ndim}"
    )
    assert time_len > 0, f"{filename} has invalid time length: {time_len}"


    if arr.ndim == 2:
        assert arr.shape[0] > 0, f"{filename} has empty feature dimension"
        assert arr.shape[1] == time_len, (
            f"{filename} time axis mismatch: arr.shape[1]={arr.shape[1]} vs time_len={time_len}"
        )


def test_subject1_predictors_have_compatible_time_axis():
    objs = {}
    for filename in EXPECTED_SUBJECT1_PREDICTORS:
        path = PREDICTOR_ROOT / filename
        objs[filename] = _load_pickle(path)

    info = {}
    for filename, obj in objs.items():
        arr, time_len, tstep, tmin = _extract_array_and_time_info(obj)
        info[filename] = {
            "arr_shape": arr.shape,
            "time_len": time_len,
            "tstep": tstep,
            "tmin": tmin,
        }

    names = list(info.keys())
    ref = info[names[0]]

    for name in names[1:]:
        cur = info[name]

        assert cur["time_len"] == ref["time_len"], (
            f"Time length mismatch between predictors: "
            f"{names[0]} has {ref['time_len']} samples, "
            f"{name} has {cur['time_len']} samples"
        )


        if ref["tstep"] is not None and cur["tstep"] is not None:
            assert cur["tstep"] == ref["tstep"], (
                f"Time step mismatch between predictors: "
                f"{names[0]} has tstep={ref['tstep']}, {name} has tstep={cur['tstep']}"
            )

        if ref["tmin"] is not None and cur["tmin"] is not None:
            assert cur["tmin"] == ref["tmin"], (
                f"Time start mismatch between predictors: "
                f"{names[0]} has tmin={ref['tmin']}, {name} has tmin={cur['tmin']}"
            )