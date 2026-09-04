"""Diagnostics for the quality of a predictive distribution."""

import numpy as np
import pytest

from cypshift.eval.uncertainty import (
    coverage,
    error_ranking,
    normalised_calibration_error,
    summarise,
)

RNG = np.random.default_rng(0)
N = 3000


def _well_calibrated(n=N):
    """Truth drawn from exactly the distribution the model states."""
    mean = RNG.normal(5.0, 1.0, n)
    std = RNG.uniform(0.2, 1.2, n)
    return mean + RNG.normal(0.0, std), mean, std


def test_a_well_calibrated_model_has_coverage_near_nominal():
    y_true, mean, std = _well_calibrated()
    for level, empirical in coverage(y_true, mean, std).items():
        assert empirical == pytest.approx(level, abs=0.03)


def test_an_overconfident_model_under_covers():
    y_true, mean, std = _well_calibrated()
    for level, empirical in coverage(y_true, mean, std / 3).items():
        assert empirical < level


def test_spread_that_tracks_error_ranks_it():
    y_true, mean, std = _well_calibrated()
    ranking = error_ranking(y_true, mean, std)
    assert ranking["spearman_std_vs_error"] > 0.3
    assert ranking["decile_error_ratio"] > 2.0


def test_a_constant_spread_reports_no_skill_rather_than_nan():
    y_true, mean, _ = _well_calibrated()
    ranking = error_ranking(y_true, mean, np.full(len(mean), 0.5))
    assert ranking["spearman_std_vs_error"] == 0.0
    assert ranking["decile_error_ratio"] == 1.0


def test_calibration_error_is_small_when_the_spread_is_right():
    y_true, mean, std = _well_calibrated()
    assert normalised_calibration_error(y_true, mean, std) < 0.15


def test_calibration_error_grows_when_the_spread_is_wrong():
    y_true, mean, std = _well_calibrated()
    honest = normalised_calibration_error(y_true, mean, std)
    overconfident = normalised_calibration_error(y_true, mean, std / 4)
    assert overconfident > honest + 0.3


def test_degenerate_spread_is_rejected_rather_than_silently_binned():
    y_true, mean, _ = _well_calibrated(50)
    with pytest.raises(ValueError, match="too degenerate"):
        normalised_calibration_error(y_true, mean, np.ones(50))


def test_summary_skips_compounds_without_a_measured_truth():
    y_true, mean, std = _well_calibrated(500)
    y_true[:100] = np.nan
    summary = summarise(y_true, mean, std)
    assert np.isfinite(list(summary.values())).all()
    assert "coverage_90" in summary and "ence" in summary
