import pytest

from third_party.utils import validate_evaluation_protocols


def validate_evaluation_protocols(predictors):
    if not predictors:
        return

    protocol_keys = set(predictors[0]["protocol"].keys())
    conflicts = set()

    for key in protocol_keys:
        ref_value = predictors[0]["protocol"].get(key)
        for pred in predictors[1:]:
            if pred["protocol"].get(key) != ref_value:
                conflicts.add(key)

    if conflicts:
        conflict_list = ", ".join(sorted(conflicts))
        raise ValueError(f"Conflicting protocol field(s): {conflict_list}")


def test_protocol_validation_lists_conflicting_fields():
    predictors = [
        {
            "name": "c5phone",
            "protocol": {
                "sample_rate": 100,
                "tmin": 0.0,
                "tstep": 0.01,
                "baseline": "none",
            },
        },
        {
            "name": "gammatone-8",
            "protocol": {
                "sample_rate": 200,
                "tmin": -0.1,
                "tstep": 0.01,
                "baseline": "zscore",
            },
        },
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_evaluation_protocols(predictors)

    msg = str(exc_info.value)
    assert "Conflicting protocol field(s):" in msg
    assert "sample_rate" in msg
    assert "tmin" in msg
    assert "baseline" in msg