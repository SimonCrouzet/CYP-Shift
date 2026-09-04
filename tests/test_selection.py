"""Deciding per endpoint whether point_estimate helps."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.decision.selection import EstimatorSelector

RNG = np.random.default_rng(0)


def _truth(n=150):
    frame = {"Molecule_Name": [f"OCNT-{i:05d}" for i in range(n)]}
    for endpoint in REGRESSION_ENDPOINTS:
        y = RNG.uniform(3.5, 7.0, n)
        frame[endpoint] = y
        frame[f"{endpoint}_conf_low"] = y - 0.2
        frame[f"{endpoint}_conf_high"] = y + 0.2
    return pd.DataFrame(frame)


def _predictions(truth, good_columns):
    """Placed predictions are accurate on `good_columns` and poor elsewhere."""
    n = len(truth)
    chosen = np.empty((n, len(REGRESSION_ENDPOINTS)))
    mean = np.empty_like(chosen)
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        y = truth[endpoint].to_numpy()
        mean[:, column] = y + RNG.normal(0, 0.6, n)
        noise = 0.15 if column in good_columns else 1.5
        chosen[:, column] = y + RNG.normal(0, noise, n)
    return chosen, mean


def test_endpoints_where_placement_helps_are_selected():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0, 3})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    assert selector.use_loss_minimising[REGRESSION_ENDPOINTS[0]]
    assert selector.use_loss_minimising[REGRESSION_ENDPOINTS[3]]


def test_endpoints_where_placement_hurts_are_rejected():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0, 3})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    assert not selector.use_loss_minimising[REGRESSION_ENDPOINTS[1]]
    assert not selector.use_loss_minimising[REGRESSION_ENDPOINTS[2]]


def test_win_rates_are_recorded_as_evidence():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    assert selector.win_rates[REGRESSION_ENDPOINTS[0]] > 0.9
    assert selector.win_rates[REGRESSION_ENDPOINTS[1]] < 0.5


def test_apply_takes_placed_values_only_where_selected():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0, 3})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    chosen = selector.apply(chosen, mean)
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        source = chosen if selector.use_loss_minimising[endpoint] else mean
        np.testing.assert_allclose(chosen[:, column], source[:, column])


def test_the_default_threshold_demands_near_unanimity():
    """A win rate just above a coin flip is not evidence at this sample size."""
    from cypshift.decision.selection import DEFAULT_WIN_RATE_THRESHOLD

    assert DEFAULT_WIN_RATE_THRESHOLD >= 0.9


def test_a_marginal_advantage_is_not_selected_by_default():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0, 3})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    marginal = [e for e, rate in selector.win_rates.items() if 0.5 < rate < 0.9]
    assert all(not selector.use_loss_minimising[e] for e in marginal)


def test_a_stricter_threshold_selects_fewer_endpoints():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0, 3})
    lenient = EstimatorSelector.fit(chosen, mean, truth, threshold=0.50)  # noqa: E501
    strict = EstimatorSelector.fit(chosen, mean, truth, threshold=0.99)
    assert sum(strict.use_loss_minimising.values()) <= sum(lenient.use_loss_minimising.values())


def test_applying_without_a_recorded_decision_is_rejected():
    truth = _truth()
    chosen, mean = _predictions(truth, good_columns={0})
    selector = EstimatorSelector.fit(chosen, mean, truth)
    selector.use_loss_minimising.pop(REGRESSION_ENDPOINTS[1])
    with pytest.raises(KeyError, match="no decision recorded"):
        selector.apply(chosen, mean)
