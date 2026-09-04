"""Verbatim reference implementation of the challenge's ranked metric.

Copied unmodified from ``evaluation/custom_scoring_functions.py`` of
https://github.com/OpenADMET/CYP-Challenge-Tutorial (Apache-2.0), whose header states
it is ported from the live scoring backend. It exists so
``tests/test_metric_matches_official.py`` can assert that our own implementation
agrees with it numerically, rather than that it looks similar.

Do not edit. If the challenge updates its scorer, replace this file wholesale and let
the differential test say whether anything moved.
"""

import numpy as np
import pandas as pd


def _resolve_bounds(
    y_true: pd.Series | np.ndarray,
    y_true_upper: pd.Series | np.ndarray | None,
    y_true_lower: pd.Series | np.ndarray | None,
    confidence_interval: float | pd.Series | np.ndarray | None,
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """Resolve upper/lower bounds from either explicit bounds or a confidence interval.

    ``confidence_interval`` and explicit bounds (``y_true_upper``/``y_true_lower``) are
    mutually exclusive. ``confidence_interval`` is treated as the full width of a band
    centred on ``y_true`` (i.e. ``y_true +/- confidence_interval / 2``).

    A side of the band that is neither given explicitly nor derivable from
    ``confidence_interval`` defaults to ``y_true`` itself — i.e. no tolerance on that
    side, since there's no information to define one. If both sides default this way
    (no bounds and no confidence_interval given at all), the band collapses to the
    point estimate on both sides, which — for callers built on top of this, like
    ``rae_soft_threshold_absolute_error`` — makes soft-thresholding a no-op and
    recovers the plain (non-thresholded) behaviour exactly.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_true_upper (pd.Series | np.ndarray | None): Optional upper bounds for true
            values.
        y_true_lower (pd.Series | np.ndarray | None): Optional lower bounds for true
            values.
        confidence_interval (float | pd.Series | np.ndarray | None): Optional
            confidence interval (full width) for true values.

    Returns:
        tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]: The resolved
            ``(y_true_lower, y_true_upper)`` bounds.

    Raises:
        ValueError: If both explicit bounds and a confidence interval are provided.

    """
    if (y_true_upper is not None or y_true_lower is not None) and confidence_interval is not None:
        raise ValueError(
            "Cannot provide both upper/lower bounds and confidence interval for soft thresholding."
        )

    if confidence_interval is not None:
        half_width = confidence_interval / 2
        y_true_upper = y_true + half_width
        y_true_lower = y_true - half_width
    else:
        if y_true_upper is None:
            y_true_upper = y_true
        if y_true_lower is None:
            y_true_lower = y_true

    return y_true_lower, y_true_upper


def rae_soft_threshold_absolute_error(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_true_upper: pd.Series | np.ndarray | None = None,
    y_true_lower: pd.Series | np.ndarray | None = None,
    confidence_interval: float | pd.Series | np.ndarray | None = None,
) -> float:
    """RAE metric for regression tasks, with soft thresholding.

    If a confidence interval is provided, the absolute error is clipped to the distance
    to the nearest bound defined by the confidence interval. The confidence interval can
    be a single float or an array of the same shape as y_true/y_pred. If upper/lower
    bounds are provided instead, the absolute error is clipped to the distance to the
    nearest bound. Explicit bounds and a confidence interval are mutually exclusive.

    A prediction that falls inside the ``[y_true_lower, y_true_upper]`` tolerance band
    contributes zero error — it is treated as indistinguishable from the true value
    given measurement uncertainty. A prediction outside the band contributes only the
    distance to the nearest edge, rather than the distance to the point estimate.

    Either ``y_true_upper`` or ``y_true_lower`` may be omitted on its own — that side
    then defaults to ``y_true`` itself (no tolerance on that side; see
    ``_resolve_bounds``). Omitting both, and no confidence_interval either, collapses
    the band to a single point on both sides, which makes soft-thresholding a no-op:
    the result is then identical to plain ``rae()``.

    The naive baseline in the denominator (a constant predictor at ``mean(y_true)``) is
    put through the same soft-thresholding as the model's predictions, so both halves
    of the ratio are computed under the same rule and RAE keeps its usual meaning:
    ``1.0`` means the model is exactly as good as always predicting the mean, under
    this tolerance-band error function.

    Args:
        y_true (pd.Series | np.ndarray): True values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        y_true_upper (pd.Series | np.ndarray | None): Optional upper bounds for true
            values. Defaults to ``y_true`` (no tolerance above) if omitted.
        y_true_lower (pd.Series | np.ndarray | None): Optional lower bounds for true
            values. Defaults to ``y_true`` (no tolerance below) if omitted.
        confidence_interval (float | pd.Series | np.ndarray | None): Optional
            confidence interval (full width) for true values.

    Returns:
        float: The relative absolute error (RAE) score, with soft-thresholded
            absolute error in both the model error and the naive-baseline error.

    Raises:
        ValueError: If both explicit bounds and a confidence interval are provided.

    """
    y_true_lower, y_true_upper = _resolve_bounds(
        y_true, y_true_upper, y_true_lower, confidence_interval
    )

    above_upper = np.clip(y_pred - y_true_upper, a_min=0, a_max=None)
    below_lower = np.clip(y_true_lower - y_pred, a_min=0, a_max=None)
    soft_abs_error = above_upper + below_lower

    mean_true = np.mean(y_true)
    baseline_above_upper = np.clip(mean_true - y_true_upper, a_min=0, a_max=None)
    baseline_below_lower = np.clip(y_true_lower - mean_true, a_min=0, a_max=None)
    soft_baseline_error = baseline_above_upper + baseline_below_lower

    return np.sum(soft_abs_error) / np.sum(soft_baseline_error)
