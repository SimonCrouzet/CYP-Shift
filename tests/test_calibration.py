"""Conformal correction of an overconfident predictive distribution."""

import numpy as np
import pytest

from cypshift.decision.calibration import ConformalCalibrator
from cypshift.eval.uncertainty import coverage
from cypshift.models.base import PredictiveDistribution

ENDPOINTS = ("CYP1A2_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")
RNG = np.random.default_rng(0)


def _overconfident(n=800, understatement=12.0):
    """A model whose spread ranks errors but understates them by a known factor."""
    truth = RNG.normal(5.0, 1.0, (n, len(ENDPOINTS)))
    honest_std = RNG.uniform(0.2, 0.8, (n, len(ENDPOINTS)))
    mean = truth + RNG.normal(0.0, honest_std)
    stated = honest_std / understatement
    members = mean[None, :, :] + RNG.normal(0.0, stated, (6, n, len(ENDPOINTS)))
    return PredictiveDistribution(members, ENDPOINTS), truth


def test_calibration_recovers_the_understated_scale():
    distribution, truth = _overconfident(understatement=12.0)
    calibrator = ConformalCalibrator.fit(distribution, truth)
    for multiplier in calibrator.multipliers.values():
        assert multiplier > 3.0


def test_calibrated_coverage_approaches_the_nominal_level():
    distribution, truth = _overconfident()
    calibrated = ConformalCalibrator.fit(distribution, truth, level=0.90).apply(distribution)
    for column, _ in enumerate(ENDPOINTS):
        before = coverage(
            truth[:, column],
            distribution.mean[:, column],
            distribution.total_std[:, column],
            levels=(0.90,),
        )[0.90]
        after = coverage(
            truth[:, column],
            calibrated.mean[:, column],
            calibrated.total_std[:, column],
            levels=(0.90,),
        )[0.90]
        assert before < 0.4
        # Tight, because a loose bound here hides a half-width-versus-deviation
        # confusion that silently inflates every interval.
        assert after == pytest.approx(0.90, abs=0.04)


def test_calibration_never_narrows_the_distribution():
    distribution, truth = _overconfident()
    calibrated = ConformalCalibrator.fit(distribution, truth).apply(distribution)
    assert (calibrated.total_std >= distribution.total_std - 1e-9).all()


def test_the_predicted_mean_is_untouched():
    distribution, truth = _overconfident()
    calibrated = ConformalCalibrator.fit(distribution, truth).apply(distribution)
    np.testing.assert_allclose(calibrated.mean, distribution.mean)


def test_the_sample_shape_survives_calibration():
    distribution, truth = _overconfident()
    calibrated = ConformalCalibrator.fit(distribution, truth).apply(distribution)
    np.testing.assert_array_equal(calibrated.samples, distribution.samples)


def test_an_endpoint_with_no_calibration_data_is_rejected():
    distribution, truth = _overconfident(n=100)
    truth[:, 0] = np.nan
    with pytest.raises(ValueError, match="no measured compounds"):
        ConformalCalibrator.fit(distribution, truth)


def test_applying_to_an_unknown_endpoint_is_rejected():
    distribution, truth = _overconfident(n=100)
    calibrator = ConformalCalibrator.fit(distribution, truth)
    calibrator.multipliers.pop(ENDPOINTS[1])
    with pytest.raises(KeyError, match="no multiplier"):
        calibrator.apply(distribution)


def test_coverage_tracks_the_requested_level():
    distribution, truth = _overconfident()
    for level in (0.80, 0.90, 0.95):
        calibrated = ConformalCalibrator.fit(distribution, truth, level=level).apply(distribution)
        achieved = coverage(
            truth[:, 0], calibrated.mean[:, 0], calibrated.total_std[:, 0], levels=(level,)
        )[level]
        assert achieved == pytest.approx(level, abs=0.05)


def test_the_floor_adapts_to_the_scale_of_the_predicted_spread():
    # A confident model and a diffuse one must not share an absolute floor, or the
    # confident one is clamped flat and loses its per-compound ranking.
    tight, truth = _overconfident(understatement=40.0)
    wide, _ = _overconfident(understatement=2.0)
    tight_floor = ConformalCalibrator.fit(tight, truth).floors[ENDPOINTS[0]]
    wide_floor = ConformalCalibrator.fit(wide, truth).floors[ENDPOINTS[0]]
    assert tight_floor < wide_floor


def test_a_higher_nominal_level_gives_a_wider_interval():
    distribution, truth = _overconfident()
    narrow = ConformalCalibrator.fit(distribution, truth, level=0.80)
    wide = ConformalCalibrator.fit(distribution, truth, level=0.95)
    for endpoint in ENDPOINTS:
        assert wide.multipliers[endpoint] > narrow.multipliers[endpoint]


def test_apply_does_not_double_count_existing_aleatoric():
    """The calibrated total must reach the conformal target, not target + prior aleatoric."""
    from scipy.stats import norm

    n = 5
    samples = np.full((4, n, len(ENDPOINTS)), 5.0)  # zero sample spread -> epistemic 0
    aleatoric = np.ones((4, n, len(ENDPOINTS)))  # already carries aleatoric variance 1
    distribution = PredictiveDistribution(samples, ENDPOINTS, aleatoric)
    assert np.allclose(distribution.total_std, 1.0)
    z = norm.ppf(0.5 + 0.90 / 2)
    # multiplier z, tiny floor => target variance == total_std**2 == 1
    calibrator = ConformalCalibrator({e: z for e in ENDPOINTS}, {e: 1e-9 for e in ENDPOINTS}, 0.90)
    out = calibrator.apply(distribution)
    assert np.allclose(out.total_std, 1.0, atol=1e-6)  # not sqrt(2) from re-adding aleatoric
