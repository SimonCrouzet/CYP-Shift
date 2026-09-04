"""Tests for leave-one-target-out evaluation."""

import numpy as np
import pandas as pd
import pytest

from cypshift.eval.transfer import (
    leave_one_target_out,
    population_mean_baseline,
    split_by_target,
)


@pytest.fixture
def observations():
    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame({"target": name, "value": rng.normal(centre, 0.5, 40)})
        for name, centre in [("CYP1A2", 5.0), ("CYP2C9", 6.0), ("CYP2D6", 4.0)]
    ]
    return pd.concat(frames, ignore_index=True)


def test_split_removes_the_held_out_enzyme_entirely(observations):
    train, test = split_by_target(observations, "CYP2C9")
    assert "CYP2C9" not in set(train["target"])
    assert set(test["target"]) == {"CYP2C9"}


def test_split_preserves_every_observation(observations):
    train, test = split_by_target(observations, "CYP2C9")
    assert len(train) + len(test) == len(observations)


def test_unknown_enzyme_is_refused(observations):
    with pytest.raises(ValueError, match="no observations"):
        split_by_target(observations, "CYP2E1")


def test_holding_out_the_only_enzyme_is_refused():
    single = pd.DataFrame({"target": ["CYP1A2"] * 5, "value": [1.0] * 5})
    with pytest.raises(ValueError, match="no training data"):
        split_by_target(single, "CYP1A2")


def test_baseline_uses_the_training_mean(observations):
    train, test = split_by_target(observations, "CYP2D6")
    expected = np.abs(test["value"] - train["value"].mean()).mean()
    assert population_mean_baseline(train, test) == pytest.approx(expected)


def test_a_model_that_predicts_the_training_mean_gains_nothing(observations):
    """The floor: reproducing the baseline must show as a gain of zero."""
    result = leave_one_target_out(
        observations,
        lambda train, test: np.full(len(test), train["value"].mean()),
        targets=("CYP1A2", "CYP2C9", "CYP2D6"),
    )
    assert np.allclose(result["gain"], 0.0, atol=1e-9)


def test_a_model_that_knows_the_held_out_centre_gains(observations):
    """A descriptor that genuinely transfers must show as a positive gain."""
    result = leave_one_target_out(
        observations,
        lambda train, test: np.full(len(test), test["value"].mean()),
        targets=("CYP1A2", "CYP2C9", "CYP2D6"),
    )
    assert (result["gain"] > 0).all()


def test_wrong_number_of_predictions_is_refused(observations):
    with pytest.raises(ValueError, match="expected"):
        leave_one_target_out(
            observations, lambda train, test: np.zeros(len(test) + 1), targets=("CYP1A2",)
        )


def test_result_is_indexed_by_held_out_enzyme(observations):
    result = leave_one_target_out(
        observations,
        lambda train, test: np.full(len(test), train["value"].mean()),
        targets=("CYP1A2", "CYP2D6"),
    )
    assert list(result.index) == ["CYP1A2", "CYP2D6"]
    assert result.loc["CYP1A2", "n_test"] == 40
