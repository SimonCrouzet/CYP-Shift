"""The scoring harness must reproduce the official challenge metric exactly."""

import numpy as np
import pytest

from cypshift.eval.metrics import (
    relative_absolute_error,
    soft_absolute_error,
    soft_threshold_rae,
)

TRUTH = np.array([4.0, 5.0, 6.0, 7.0])
LOWER = np.array([3.8, 4.9, 5.7, 6.5])
UPPER = np.array([4.2, 5.1, 6.3, 7.5])


def test_prediction_inside_the_band_costs_nothing():
    assert soft_absolute_error(TRUTH, LOWER, UPPER).sum() == 0.0


def test_prediction_exactly_on_a_bound_costs_nothing():
    assert soft_absolute_error(LOWER, LOWER, UPPER).sum() == 0.0
    assert soft_absolute_error(UPPER, LOWER, UPPER).sum() == 0.0


def test_error_is_distance_to_the_nearest_bound_not_to_the_point_estimate():
    # 1.0 above every upper bound, and 1.0 below every lower bound.
    np.testing.assert_allclose(soft_absolute_error(UPPER + 1.0, LOWER, UPPER), 1.0)
    np.testing.assert_allclose(soft_absolute_error(LOWER - 1.0, LOWER, UPPER), 1.0)


def test_only_one_side_can_fire():
    predictions = np.array([0.0, 100.0, 6.0, 7.0])  # far low, far high, then two inside
    errors = soft_absolute_error(predictions, LOWER, UPPER)
    above = np.clip(predictions - UPPER, 0, None)
    below = np.clip(LOWER - predictions, 0, None)
    assert not np.any((above > 0) & (below > 0))
    np.testing.assert_allclose(errors, above + below)
    assert errors[2] == 0.0 and errors[3] == 0.0


def test_mean_predictor_scores_exactly_one():
    predictions = np.full_like(TRUTH, TRUTH.mean())
    assert soft_threshold_rae(TRUTH, predictions, LOWER, UPPER) == pytest.approx(1.0)


def test_perfect_prediction_scores_zero():
    assert soft_threshold_rae(TRUTH, TRUTH, LOWER, UPPER) == 0.0


def test_collapsing_the_band_recovers_plain_rae():
    strae = soft_threshold_rae(TRUTH, TRUTH + 0.5, TRUTH, TRUTH)
    assert strae == pytest.approx(relative_absolute_error(TRUTH, TRUTH + 0.5))


def test_wider_bands_never_increase_the_numerator():
    predictions = UPPER + 1.0
    narrow = soft_absolute_error(predictions, LOWER, UPPER).sum()
    wide = soft_absolute_error(predictions, LOWER - 1, UPPER + 1).sum()
    assert wide <= narrow


def test_undefined_when_the_mean_predictor_is_never_penalised():
    # A interval spanning everything leaves the baseline with zero error.
    with pytest.raises(ZeroDivisionError):
        soft_threshold_rae(TRUTH, TRUTH, TRUTH - 10, TRUTH + 10)


def test_matches_a_hand_computed_example():
    truth = np.array([2.0, 6.0])
    lower, upper = np.array([1.5, 5.5]), np.array([2.5, 6.5])
    predictions = np.array([4.0, 6.0])  # 1.5 above the first interval, inside the second
    # baseline is mean(truth) = 4.0: 1.5 above the first interval, 1.5 below the second
    assert soft_threshold_rae(truth, predictions, lower, upper) == pytest.approx(1.5 / 3.0)
