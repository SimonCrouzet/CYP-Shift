"""Is a model's predictive distribution any good?

The point-estimate step optimises against whatever posterior it is handed, so a
confidently wrong distribution makes it commit harder to a wrong answer. Three
questions, and they are not the same question:

*Does the spread rank errors?*
    ``error_ranking`` -- Spearman correlation between predicted standard deviation
    and realised absolute error, plus the ratio of mean error in the least-confident
    decile to the most-confident. A model can rank errors well while being badly
    wrong about their absolute size.

*Is the spread the right size?*
    ``normalised_calibration_error`` -- binned comparison of predicted deviation
    against realised error. This is what conformal rescaling corrects.

*Do the stated intervals contain the truth as often as claimed?*
    ``coverage`` -- the end-to-end check.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, spearmanr

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = (0.80, 0.90, 0.95)


def error_ranking(
    y_true: NDArray[np.float64], mean: NDArray[np.float64], std: NDArray[np.float64]
) -> dict[str, float]:
    """How well the predicted spread orders compounds by how wrong they turn out to be.

    A model reporting the same spread for every compound cannot rank anything, and the
    rank correlation is then undefined rather than merely small. That case reports the
    no-skill values, 0.0 correlation and a decile ratio of 1.0, rather than NaN.
    """
    absolute_error = np.abs(y_true - mean)
    if np.ptp(std) == 0:
        logger.warning("predicted spread is constant; it cannot rank errors")
        return {"spearman_std_vs_error": 0.0, "decile_error_ratio": 1.0}
    order = np.argsort(std)
    decile = max(1, len(std) // 10)
    least_confident = absolute_error[order][-decile:].mean()
    most_confident = absolute_error[order][:decile].mean()
    return {
        "spearman_std_vs_error": float(spearmanr(std, absolute_error).statistic),
        "decile_error_ratio": float(least_confident / max(most_confident, 1e-9)),
    }


def normalised_calibration_error(
    y_true: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Expected normalised calibration error: how far off the spread is in absolute size.

    Compounds are binned by predicted standard deviation; within each bin the
    predicted deviation is compared with the realised root mean squared error. Zero is
    perfect. A value of 0.4 means the stated spread is wrong by about 40%.
    """
    absolute_error = np.abs(y_true - mean)
    edges = np.unique(np.quantile(std, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("predicted standard deviations are too degenerate to bin")
    assignment = np.clip(np.digitize(std, edges[1:-1]), 0, len(edges) - 2)
    errors = []
    for index in range(len(edges) - 1):
        in_bin = assignment == index
        if not in_bin.any():
            continue
        predicted = np.sqrt(np.mean(std[in_bin] ** 2))
        realised = np.sqrt(np.mean(absolute_error[in_bin] ** 2))
        errors.append(abs(predicted - realised) / max(realised, 1e-9))
    return float(np.mean(errors))


def coverage(
    y_true: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    levels: tuple[float, ...] = DEFAULT_LEVELS,
) -> dict[float, float]:
    """Fraction of compounds whose truth falls inside the stated Gaussian interval."""
    results = {}
    for level in levels:
        half_width = norm.ppf(0.5 + level / 2) * std
        inside = np.abs(y_true - mean) <= half_width
        results[level] = float(inside.mean())
    return results


def summarise(
    y_true: NDArray[np.float64], mean: NDArray[np.float64], std: NDArray[np.float64]
) -> dict[str, float]:
    """All three diagnostics at once, on the compounds with a measured truth."""
    measured = np.isfinite(y_true)
    y_true, mean, std = y_true[measured], mean[measured], std[measured]
    summary = {
        **error_ranking(y_true, mean, std),
        "ence": normalised_calibration_error(y_true, mean, std),
        "median_predicted_std": float(np.median(std)),
        "median_absolute_error": float(np.median(np.abs(y_true - mean))),
    }
    summary.update({f"coverage_{int(k * 100)}": v for k, v in coverage(y_true, mean, std).items()})
    return summary


def conditioning_is_additive(
    predictions_by_assay: dict[str, NDArray[np.float64]], tolerance: float = 0.05
) -> dict[str, float]:
    """Test whether an assay changes more than the intercept.

    Concatenating a one-hot at a network's input can only shift the output, so a model
    can satisfy assay conditioning entirely by learning a per-assay constant. That is a
    degenerate solution: the assay label buys an intercept and the assay's data
    contributes nothing to the chemistry.

    If the conditioning is doing real work, the difference between two assays'
    predictions varies across compounds. If it is only an intercept, that difference is
    the same number everywhere.

    Args:
        predictions_by_assay: Predictions for the *same compounds* under each assay,
            each of shape ``(n_compounds,)``.
        tolerance: Spread below which the difference counts as constant, in log units.

    Returns:
        For each assay pair, the standard deviation of the per-compound difference, and
        a flag saying whether it is indistinguishable from a constant offset.

    Raises:
        ValueError: If fewer than two assays are supplied.
    """
    names = sorted(predictions_by_assay)
    if len(names) < 2:
        raise ValueError("need at least two assays to compare")
    summary: dict[str, float] = {}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            difference = predictions_by_assay[first] - predictions_by_assay[second]
            spread = float(np.std(difference))
            summary[f"{first}_vs_{second}_offset"] = float(np.mean(difference))
            summary[f"{first}_vs_{second}_spread"] = spread
            summary[f"{first}_vs_{second}_is_constant"] = float(spread < tolerance)
            if spread < tolerance:
                logger.warning(
                    "%s vs %s differ by a near-constant %.2f (spread %.3f): the "
                    "conditioning is acting as an intercept only",
                    first,
                    second,
                    np.mean(difference),
                    spread,
                )
    return summary
