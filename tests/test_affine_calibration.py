"""The affine recalibration map fit by scripts/calibrate.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from calibrate import affine_for_endpoint  # noqa: E402


def _compressed(n=4000):
    """Truth, and a prediction that ranks it well but under-spreads it."""
    rng = np.random.default_rng(0)
    truth = rng.normal(5.0, 1.2, n)
    pred = 0.5 * truth + rng.normal(0.0, 0.8, n)
    band = np.full(n, 0.3)
    return truth, pred, truth - band, truth + band


def test_variance_match_rho_targets_the_r2_optimal_spread():
    """variance_match_rho sets the slope to rho * sigma_true / sigma_pred."""
    truth, pred, low, high = _compressed()
    slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match_rho", 0.5)
    rho = np.corrcoef(truth, pred)[0, 1]
    assert slope == pytest.approx(rho * truth.std() / pred.std(), rel=1e-6)


def test_rho_scaling_is_less_aggressive_than_full_variance_match():
    """Because rho < 1, de-shrinking to rho*sigma never overshoots the full-sigma target."""
    truth, pred, low, high = _compressed()
    rho_slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match_rho", 0.5)
    full_slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match", 0.5)
    assert 0.0 < rho_slope < full_slope
