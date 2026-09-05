"""Tests for cross-model disagreement as an epistemic uncertainty estimate."""

import numpy as np
import pytest

from cypshift.eval.disagreement import (
    calibrate_width,
    cross_model_spread,
    gaussian_width,
)


def test_spread_is_zero_when_models_agree():
    same = np.array([[5.0, 6.0], [4.0, 3.0]])
    assert not cross_model_spread({"a": same, "b": same.copy()}).any()


def test_spread_grows_with_disagreement():
    a = np.zeros((3, 2))
    near = cross_model_spread({"a": a, "b": a + 0.1})
    far = cross_model_spread({"a": a, "b": a + 2.0})
    assert (far > near).all()


def test_a_single_model_is_refused():
    """A spread over one model is zero everywhere and would masquerade as confidence."""
    with pytest.raises(ValueError, match="at least two"):
        cross_model_spread({"only": np.zeros((3, 2))})


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="differ in shape"):
        cross_model_spread({"a": np.zeros((3, 2)), "b": np.zeros((3, 3))})


def test_spread_keeps_the_prediction_shape():
    shape = (5, 4)
    out = cross_model_spread({"a": np.zeros(shape), "b": np.ones(shape)})
    assert out.shape == shape


def test_calibrated_interval_reaches_the_requested_coverage():
    """The scale should leave about `level` of the calibration residuals inside."""
    rng = np.random.default_rng(0)
    spread = rng.uniform(0.2, 1.0, 2000)
    residual = rng.normal(0.0, 1.0, 2000) * spread
    fitted = calibrate_width(spread, residual, level=0.9)
    lower, upper = fitted.interval(np.zeros_like(spread))
    inside = (residual >= lower) & (residual <= upper)
    assert inside.mean() == pytest.approx(0.9, abs=0.03)


def test_calibration_uses_the_quantile_directly_not_as_a_sigma():
    """Multiplying the fitted quantile by the normal quantile again overshoots
    coverage; this guards against that regression."""
    rng = np.random.default_rng(1)
    spread = rng.uniform(0.2, 1.0, 3000)
    residual = rng.normal(0, 1, 3000) * spread
    fitted = calibrate_width(spread, residual, level=0.8)
    lower, upper = fitted.interval(np.zeros_like(spread))
    coverage = ((residual >= lower) & (residual <= upper)).mean()
    assert coverage < 0.9  # not blown out to ~0.95+


def test_the_floor_is_a_quantile_of_spread_not_a_constant():
    """A fixed floor flattens a narrow distribution; the floor should scale with it."""
    narrow = calibrate_width(np.full(100, 0.03), np.full(100, 0.03))
    assert narrow.floor < 0.05


def test_mismatched_calibration_lengths_are_refused():
    with pytest.raises(ValueError, match="differ"):
        calibrate_width(np.zeros(5), np.zeros(4))


def test_gaussian_width_scales_with_the_level():
    sigma = np.array([1.0, 2.0])
    assert (gaussian_width(sigma, 0.99) > gaussian_width(sigma, 0.90)).all()


def test_interval_is_symmetric_around_the_centre():
    fitted = calibrate_width(np.full(50, 0.5), np.full(50, 0.4))
    centre = np.array([5.0, 6.0, 4.0])
    lower, upper = fitted.interval(centre, spread=np.full(3, 0.5))
    assert np.allclose(upper - centre, centre - lower)


def test_interval_applies_the_fitted_scale_to_new_compounds():
    """The fitted scale must be usable on compounds outside the calibration set, whose
    own disagreement is passed in."""
    fitted = calibrate_width(np.full(20, 0.5), np.full(20, 0.5))
    centre = np.zeros(2)
    narrow = fitted.interval(centre, spread=np.array([0.1, 0.1]))
    wide = fitted.interval(centre, spread=np.array([2.0, 2.0]))
    assert (wide[1] - wide[0] > narrow[1] - narrow[0]).all()
