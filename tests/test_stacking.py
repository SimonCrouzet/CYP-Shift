"""The hinge-ridge correction: what it fits, how it is scored, and what it does with noise."""

import numpy as np
import pytest

from cypshift.eval.stacking import (
    apply_stack,
    cross_validated_stack,
    fit_stack,
    fit_weights,
    hinge,
    restore_moments,
    standardise,
)


def _problem(n: int = 600, seed: int = 0):
    """Predictions whose error follows one column (``signal``) and not another (``noise``)."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(5.0, 1.0, n)
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    base = truth + 0.8 * signal
    low, high = truth - 0.1, truth + 0.1
    folds = rng.integers(0, 5, n)
    return base, signal, noise, low, high, folds


def test_the_hinge_is_the_distance_outside_the_interval_and_zero_inside():
    prediction = np.array([4.2, 5.0, 5.7, 6.0, 7.1])
    low, high = np.full(5, 5.0), np.full(5, 6.0)
    assert hinge(prediction, low, high).tolist() == pytest.approx([0.8, 0.0, 0.0, 0.0, 1.1])


def test_the_fitted_weight_removes_the_error_a_column_explains():
    base, signal, _, low, high, _ = _problem()
    z = ((signal - signal.mean()) / signal.std()).reshape(-1, 1)
    weights = fit_weights(z, base, low, high, l2=1e-4)
    assert weights[0] == pytest.approx(-0.8 * signal.std(), abs=0.05)
    assert hinge(base + z @ weights, low, high).mean() < 0.1 * hinge(base, low, high).mean()


def test_a_heavy_penalty_keeps_the_weights_at_zero():
    base, signal, _, low, high, _ = _problem()
    weights = fit_weights(signal.reshape(-1, 1), base, low, high, l2=1e4)
    assert abs(weights[0]) < 1e-3


def test_weights_can_be_negative():
    base, signal, _, low, high, folds = _problem()
    stack = fit_stack(base, signal.reshape(-1, 1), low, high, folds)
    assert stack.weights[0] < -0.3


def test_the_cross_validated_correction_helps_with_a_real_column_and_not_with_noise():
    base, signal, noise, low, high, folds = _problem()
    columns = np.column_stack([signal, noise])
    corrected, chosen = cross_validated_stack(base, columns, low, high, folds)
    plain = hinge(base, low, high).mean()
    assert hinge(corrected, low, high).mean() < 0.3 * plain
    assert len(chosen) == 5
    only_noise, _ = cross_validated_stack(base, noise.reshape(-1, 1), low, high, folds)
    assert hinge(only_noise, low, high).mean() <= 1.02 * plain


def test_a_compound_is_corrected_by_weights_fitted_without_its_fold():
    base, signal, _, low, high, folds = _problem(n=200)
    columns = signal.reshape(-1, 1)
    corrected, _ = cross_validated_stack(base, columns, low, high, folds)
    held = folds == 0
    outside = fit_stack(base[~held], columns[~held], low[~held], high[~held], folds[~held])
    assert np.allclose(corrected[held], apply_stack(outside, base[held], columns[held]))


def test_a_missing_column_value_leaves_the_prediction_alone():
    base, signal, _, low, high, folds = _problem(n=100)
    stack = fit_stack(base, signal.reshape(-1, 1), low, high, folds)
    missing = np.array([[np.nan]])
    assert apply_stack(stack, np.array([5.0]), missing)[0] == 5.0
    assert standardise(missing, stack.centre, stack.scale)[0, 0] == 0.0


def test_restoring_the_moments_keeps_the_order_and_borrows_level_and_spread():
    rng = np.random.default_rng(1)
    raw, reference = rng.normal(0.0, 5.0, 50), rng.normal(4.0, 0.7, 50)
    restored = restore_moments(raw, reference)
    assert restored.mean() == pytest.approx(reference.mean())
    assert restored.std() == pytest.approx(reference.std())
    assert (np.argsort(restored) == np.argsort(raw)).all()
    assert restore_moments(np.full(5, 3.0), reference) == pytest.approx(
        np.full(5, reference.mean())
    )
