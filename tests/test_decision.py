"""Band-width estimation and expected-loss placement."""

import numpy as np
import pytest

from cypshift.decision.band import BandWidthModel
from cypshift.decision.placement import expected_soft_loss, optimal_placement, place
from cypshift.models.base import PredictiveDistribution

ENDPOINTS = ("CYP1A2_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")


def _observed_intervals(n=600, seed=0):
    """Truth with realistically activity-dependent bands: wide below 4, tight above."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(2.0, 7.5, n)
    half = np.where(y < 4.0, 1.2, 0.2)
    return y, y - half, y + half


def test_band_model_recovers_the_activity_dependence():
    y, low, high = _observed_intervals()
    model = BandWidthModel.fit(y, low, high)
    inactive_lower, _ = model.predict(np.array([2.5]))
    active_lower, _ = model.predict(np.array([6.5]))
    assert inactive_lower[0] > 3 * active_lower[0]


def test_band_model_needs_enough_observations():
    with pytest.raises(ValueError, match="at least 12"):
        BandWidthModel.fit(np.arange(5.0), np.arange(5.0) - 1, np.arange(5.0) + 1)


def test_band_model_ignores_incomplete_rows():
    y, low, high = _observed_intervals()
    y[:20] = np.nan
    assert BandWidthModel.fit(y, low, high) is not None


def test_band_model_does_not_extrapolate_to_negative_widths():
    y, low, high = _observed_intervals()
    model = BandWidthModel.fit(y, low, high)
    lower, upper = model.predict(np.array([-50.0, 50.0]))
    assert (lower > 0).all() and (upper > 0).all()


def test_exact_placement_matches_a_brute_force_scan():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.normal(5.0, 1.2, 300)
        lower, upper = y - rng.uniform(0.1, 1.0), y + rng.uniform(0.1, 1.0)
        exact = optimal_placement(lower, upper)
        grid = np.linspace(lower.min() - 1, upper.max() + 1, 4000)
        scanned = grid[np.argmin(expected_soft_loss(grid, lower, upper))]
        loss_exact = expected_soft_loss(np.array([exact]), lower, upper)[0]
        loss_scanned = expected_soft_loss(np.array([scanned]), lower, upper)[0]
        # The exact solution can only beat a finite scan, never lose to it.
        assert loss_exact <= loss_scanned + 1e-9


def test_placement_lands_inside_a_single_wide_band():
    lower, upper = np.full(50, 3.0), np.full(50, 6.0)
    assert 3.0 <= optimal_placement(lower, upper) <= 6.0


def test_placement_commits_to_a_mode_rather_than_the_mean():
    # Two thirds inactive near 3, one third active near 7. The mean sits near 4.3,
    # in the empty middle; the loss-minimising point should not.
    values = np.concatenate([np.full(200, 3.0), np.full(100, 7.0)])
    lower, upper = values - 0.2, values + 0.2
    placement = optimal_placement(lower, upper)
    assert abs(placement - values.mean()) > 0.5
    assert min(abs(placement - 3.0), abs(placement - 7.0)) < 0.5


def test_wider_bands_pull_the_placement_no_further_than_the_bounds():
    y = np.random.default_rng(2).normal(5.0, 0.5, 200)
    placement = optimal_placement(y - 2.0, y + 2.0)
    assert (y - 2.0).min() <= placement <= (y + 2.0).max()


def test_place_returns_one_value_per_compound_and_endpoint():
    y, low, high = _observed_intervals()
    models = {endpoint: BandWidthModel.fit(y, low, high) for endpoint in ENDPOINTS}
    rng = np.random.default_rng(4)
    distribution = PredictiveDistribution(rng.normal(5.0, 0.5, (6, 9, 2)), ENDPOINTS)
    placements = place(distribution, models, n_draws=100)
    assert placements.shape == (9, 2)
    assert np.isfinite(placements).all()


def test_place_rejects_a_missing_band_model():
    y, low, high = _observed_intervals()
    distribution = PredictiveDistribution(np.zeros((2, 3, 2)), ENDPOINTS)
    with pytest.raises(KeyError, match="no band model"):
        place(distribution, {ENDPOINTS[0]: BandWidthModel.fit(y, low, high)})
