"""An optimization-free or reduced run must never masquerade as notebook reproduction."""
import pytest

from train import train_model


@pytest.mark.parametrize("parameters", [
    {"mode": "smoke", "n_trials": 0},
    {"mode": "smoke", "max_estimators": 0},
    {"mode": "notebook", "n_trials": 2},
    {"mode": "notebook", "max_estimators": 200},
])
def test_invalid_training_conditions_fail_before_reading_data(tmp_path, parameters):
    artifact = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        train_model(tmp_path / "absent_input.csv", artifact_path=artifact, **parameters)
    assert not artifact.exists()
