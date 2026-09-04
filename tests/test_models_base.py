"""The predictive distribution carried between routes and the decision layer."""

import numpy as np
import pytest

from cypshift.models.base import PredictiveDistribution

ENDPOINTS = ("CYP1A2_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")


def _distribution(n_members=5, n_compounds=4, with_aleatoric=False):
    rng = np.random.default_rng(0)
    samples = rng.normal(5.0, 0.4, (n_members, n_compounds, len(ENDPOINTS)))
    variance = np.full_like(samples, 0.09) if with_aleatoric else None
    return PredictiveDistribution(samples, ENDPOINTS, variance)


def test_shape_mismatch_between_samples_and_endpoints_is_rejected():
    with pytest.raises(ValueError, match="endpoint columns"):
        PredictiveDistribution(np.zeros((2, 3, 5)), ENDPOINTS)


def test_two_dimensional_samples_are_rejected():
    with pytest.raises(ValueError, match="must be"):
        PredictiveDistribution(np.zeros((3, 2)), ENDPOINTS)


def test_mismatched_aleatoric_shape_is_rejected():
    with pytest.raises(ValueError, match="aleatoric_variance"):
        PredictiveDistribution(np.zeros((2, 3, 2)), ENDPOINTS, np.zeros((2, 3)))


def test_mean_collapses_the_member_axis():
    distribution = _distribution()
    assert distribution.mean.shape == (4, len(ENDPOINTS))
    np.testing.assert_allclose(distribution.mean, distribution.samples.mean(axis=0))


def test_a_single_member_has_no_epistemic_spread():
    distribution = _distribution(n_members=1)
    np.testing.assert_allclose(distribution.epistemic_variance, 0.0)


def test_total_variance_adds_both_sources():
    distribution = _distribution(with_aleatoric=True)
    expected = np.sqrt(distribution.epistemic_variance + 0.09)
    np.testing.assert_allclose(distribution.total_std, expected)


def test_total_variance_is_epistemic_only_when_no_noise_is_reported():
    distribution = _distribution()
    np.testing.assert_allclose(distribution.total_std**2, distribution.epistemic_variance)


def test_draws_are_reproducible_and_correctly_shaped():
    distribution = _distribution(with_aleatoric=True)
    first = distribution.draw(50, seed=3)
    second = distribution.draw(50, seed=3)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (50, 4, len(ENDPOINTS))


def test_draws_widen_when_aleatoric_noise_is_present():
    without = _distribution().draw(4000, seed=1).std()
    with_noise = _distribution(with_aleatoric=True).draw(4000, seed=1).std()
    assert with_noise > without


def test_narrowing_to_one_endpoint_keeps_the_structure():
    single = _distribution(with_aleatoric=True).endpoint(ENDPOINTS[1])
    assert single.endpoints == (ENDPOINTS[1],)
    assert single.samples.shape == (5, 4, 1)
    assert single.aleatoric_variance.shape == (5, 4, 1)


def test_quantiles_are_ordered():
    distribution = _distribution()
    assert (distribution.quantile(0.1) <= distribution.quantile(0.9)).all()
