"""Deciding per endpoint whether placement helps."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.decision.selection import PlacementSelector

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
    placed = np.empty((n, len(REGRESSION_ENDPOINTS)))
    mean = np.empty_like(placed)
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        y = truth[endpoint].to_numpy()
        mean[:, column] = y + RNG.normal(0, 0.6, n)
        noise = 0.15 if column in good_columns else 1.5
        placed[:, column] = y + RNG.normal(0, noise, n)
    return placed, mean


def test_endpoints_where_placement_helps_are_selected():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0, 3})
    selector = PlacementSelector.fit(placed, mean, truth)
    assert selector.place_endpoint[REGRESSION_ENDPOINTS[0]]
    assert selector.place_endpoint[REGRESSION_ENDPOINTS[3]]


def test_endpoints_where_placement_hurts_are_rejected():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0, 3})
    selector = PlacementSelector.fit(placed, mean, truth)
    assert not selector.place_endpoint[REGRESSION_ENDPOINTS[1]]
    assert not selector.place_endpoint[REGRESSION_ENDPOINTS[2]]


def test_win_rates_are_recorded_as_evidence():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0})
    selector = PlacementSelector.fit(placed, mean, truth)
    assert selector.win_rates[REGRESSION_ENDPOINTS[0]] > 0.9
    assert selector.win_rates[REGRESSION_ENDPOINTS[1]] < 0.5


def test_apply_takes_placed_values_only_where_selected():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0, 3})
    selector = PlacementSelector.fit(placed, mean, truth)
    chosen = selector.apply(placed, mean)
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        source = placed if selector.place_endpoint[endpoint] else mean
        np.testing.assert_allclose(chosen[:, column], source[:, column])


def test_a_stricter_threshold_selects_fewer_endpoints():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0, 3})
    lenient = PlacementSelector.fit(placed, mean, truth, threshold=0.50)
    strict = PlacementSelector.fit(placed, mean, truth, threshold=0.99)
    assert sum(strict.place_endpoint.values()) <= sum(lenient.place_endpoint.values())


def test_applying_without_a_recorded_decision_is_rejected():
    truth = _truth()
    placed, mean = _predictions(truth, good_columns={0})
    selector = PlacementSelector.fit(placed, mean, truth)
    selector.place_endpoint.pop(REGRESSION_ENDPOINTS[1])
    with pytest.raises(KeyError, match="no decision recorded"):
        selector.apply(placed, mean)
