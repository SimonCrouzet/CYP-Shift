"""Regression metrics for the CYP challenge, reproducing the official scoring code.

The ranked metric is a *soft-threshold* L1 loss: a prediction falling anywhere inside the
95% credible interval of the fitted dose-response curve contributes exactly zero
error, and one outside contributes only its distance to the nearest bound. Both the
model error and the naive-baseline error in the denominator go through the same
soft-thresholding, so 1.0 keeps its usual meaning of "no better than predicting the
mean".

Ported from ``evaluation/custom_scoring_functions.py`` of
https://github.com/OpenADMET/CYP-Challenge-Tutorial, which the organisers state is
ported from the live scoring backend. Two further RAE variants exist in that file
(``rae_weight_below_threshold`` and ``rae_inverse_confidence_weighting``); neither is
used by the leaderboard and neither is reproduced here.
"""

import logging
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def _as_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float)


def soft_absolute_error(
    y_pred: ArrayLike,
    y_true_lower: ArrayLike,
    y_true_upper: ArrayLike,
) -> NDArray[np.float64]:
    """Per-compound distance from a prediction to the nearest credible-interval bound.

    Zero anywhere inside the closed interval ``[y_true_lower, y_true_upper]``, including
    exactly on a bound. At most one of the two terms can be non-zero.

    Args:
        y_pred: Predicted pIC50 values.
        y_true_lower: Lower credible-interval bounds (the ``_conf_low`` column).
        y_true_upper: Upper credible-interval bounds (the ``_conf_high`` column).

    Returns:
        Unreduced per-compound error, same shape as ``y_pred``.
    """
    pred = _as_array(y_pred)
    above_upper = np.clip(pred - _as_array(y_true_upper), a_min=0, a_max=None)
    below_lower = np.clip(_as_array(y_true_lower) - pred, a_min=0, a_max=None)
    return above_upper + below_lower


def soft_threshold_rae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_true_lower: ArrayLike,
    y_true_upper: ArrayLike,
) -> float:
    """Soft-threshold relative absolute error (ST-RAE). Lower is better.

    The denominator is the same soft-thresholded error committed by a constant
    predictor at ``mean(y_true)``, so it does not depend on the submission: minimising
    ST-RAE is exactly minimising the summed soft-threshold L1 error.

    Args:
        y_true: True pIC50 point estimates. Used only for the baseline mean.
        y_pred: Predicted pIC50 values.
        y_true_lower: Lower credible-interval bounds.
        y_true_upper: Upper credible-interval bounds.

    Returns:
        ST-RAE. 1.0 means no better than predicting the mean of ``y_true``.

    Raises:
        ZeroDivisionError: If the baseline commits no error, which makes the ratio
            undefined. The official implementation would emit a bare ``inf`` here.
    """
    truth = _as_array(y_true)
    model_error = soft_absolute_error(y_pred, y_true_lower, y_true_upper).sum()
    baseline_error = soft_absolute_error(
        np.full_like(truth, truth.mean()), y_true_lower, y_true_upper
    ).sum()
    if baseline_error == 0:
        raise ZeroDivisionError(
            "the mean predictor falls inside every credible interval, so ST-RAE is undefined"
        )
    return float(model_error / baseline_error)


def relative_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Plain RAE, the zero-tolerance limit of :func:`soft_threshold_rae`."""
    truth, pred = _as_array(y_true), _as_array(y_pred)
    return float(np.abs(truth - pred).sum() / np.abs(truth - truth.mean()).sum())


def _rank_correlation(statistic: object) -> object:
    """Wrap a scipy rank correlation, silencing its constant-input warning.

    A bootstrap sample can legitimately draw a constant vector, which leaves the
    correlation undefined. That case is expected and handled by
    :data:`METRIC_NAN_FALLBACK`, so the warning carries no information.
    """

    def compute(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="An input array is constant")
            return float(statistic(y_true, y_pred).statistic)

    return compute


#: Secondary leaderboard metrics, in the order the challenge reports them. Each takes
#: ``(y_true, y_pred)``; ST-RAE is handled separately because it needs the bounds.
SECONDARY_METRICS = {
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "Spearman_R": _rank_correlation(spearmanr),
    "Kendall_Tau": _rank_correlation(kendalltau),
}

#: Rank correlations are undefined when a bootstrap sample has zero variance in either
#: vector. The official code substitutes the no-correlation value rather than failing.
METRIC_NAN_FALLBACK = {"Spearman_R": 0.0, "Kendall_Tau": 0.0}
