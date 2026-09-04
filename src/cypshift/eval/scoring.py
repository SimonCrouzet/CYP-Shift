"""Bootstrapped, macro-averaged scoring of a regression submission.

Reproduces ``evaluation/evaluate_predictions.py`` and ``evaluation/utils.py`` of the
official tutorial repository. Three details matter and are easy to get wrong:

* The resampling seed is fixed, so every submission is scored on identical bootstrap
  samples and comparisons between submissions are paired.
* Each endpoint is resampled over *its own* row count, independently of the others.
* The macro average is a plain mean over endpoints taken **inside** each bootstrap
  sample, which is what gives the macro row its own bootstrap distribution.
"""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CREDIBLE_LOWER_SUFFIX,
    CREDIBLE_UPPER_SUFFIX,
    MACRO_LABEL,
    REGRESSION_ENDPOINTS,
)
from cypshift.eval.metrics import METRIC_NAN_FALLBACK, SECONDARY_METRICS, soft_threshold_rae

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "ST-RAE"


@lru_cache(maxsize=8)
def bootstrap_indices(n_rows: int, n_repeats: int = BOOTSTRAP_SAMPLES) -> NDArray[np.int64]:
    """Seeded resampling indices, shape ``(n_repeats, n_rows)``.

    The seed is fixed at :data:`cypshift.config.BOOTSTRAP_SEED` so that scores are
    reproducible and differences between submissions are paired across samples.
    """
    rng = np.random.default_rng(seed=BOOTSTRAP_SEED)
    return rng.choice(n_rows, size=(n_repeats, n_rows), replace=True)


def score_endpoint(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_true_lower: NDArray[np.float64],
    y_true_upper: NDArray[np.float64],
) -> dict[str, float]:
    """All leaderboard metrics for one endpoint, without bootstrapping."""
    scores = {PRIMARY_METRIC: soft_threshold_rae(y_true, y_pred, y_true_lower, y_true_upper)}
    for name, metric in SECONDARY_METRICS.items():
        value = float(metric(y_true, y_pred))
        if not np.isfinite(value):
            if name not in METRIC_NAN_FALLBACK:
                raise ValueError(f"{name} is not finite and has no defined fallback")
            value = METRIC_NAN_FALLBACK[name]
        scores[name] = value
    return scores


def bootstrap_endpoint(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    y_true_lower: NDArray[np.float64],
    y_true_upper: NDArray[np.float64],
    n_repeats: int = BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    """One row of metrics per bootstrap sample for a single endpoint."""
    samples = bootstrap_indices(len(y_true), n_repeats)
    rows = [
        score_endpoint(y_true[idx], y_pred[idx], y_true_lower[idx], y_true_upper[idx])
        for idx in samples
    ]
    return pd.DataFrame(rows)


def _bootstrap_all_endpoints(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    key: str,
    n_repeats: int,
) -> dict[str, pd.DataFrame]:
    """Per-sample metrics for every endpoint plus the macro row."""
    merged = truth.merge(predictions, on=key, suffixes=("", "_pred"), validate="one_to_one")
    per_sample: dict[str, pd.DataFrame] = {}
    for endpoint in REGRESSION_ENDPOINTS:
        measured = merged[endpoint].notna()
        subset = merged.loc[measured]
        predicted = subset[f"{endpoint}_pred"].to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{endpoint}: predictions must be finite where truth exists")
        logger.info("scoring %s on %d compounds", endpoint, len(subset))
        per_sample[endpoint] = bootstrap_endpoint(
            subset[endpoint].to_numpy(dtype=float),
            predicted,
            subset[f"{endpoint}{CREDIBLE_LOWER_SUFFIX}"].to_numpy(dtype=float),
            subset[f"{endpoint}{CREDIBLE_UPPER_SUFFIX}"].to_numpy(dtype=float),
            n_repeats,
        )
    # The macro row is averaged within each bootstrap sample, not after aggregation.
    per_sample[MACRO_LABEL] = (
        pd.concat(per_sample.values(), keys=range(len(per_sample))).groupby(level=1).mean()
    )
    return per_sample


def score_submission(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    key: str = "Molecule_Name",
    n_repeats: int = BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    """Score a regression submission the way the leaderboard does.

    Compounds whose truth is missing are dropped for that endpoint only, so each
    endpoint bootstraps over its own row count. A missing *prediction* where truth
    exists is an error, matching the official validator.

    Args:
        predictions: One row per compound, with ``key`` and the four endpoint columns.
        truth: Ground truth, with ``key``, the endpoint columns and their
            ``_conf_low`` / ``_conf_high`` companions.
        key: Column identifying a compound in both frames.
        n_repeats: Bootstrap samples. The leaderboard uses 1000.

    Returns:
        One row per endpoint plus a macro row labelled ``MA``, with a ``_mean`` and
        ``_std`` column per metric taken over the bootstrap distribution.
    """
    per_sample = _bootstrap_all_endpoints(predictions, truth, key, n_repeats)

    summary = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                **{f"{m}_mean": frame[m].mean() for m in frame.columns},
                **{f"{m}_std": frame[m].std(ddof=1) for m in frame.columns},
            }
            for endpoint, frame in per_sample.items()
        ]
    )
    return summary.set_index("endpoint")


def paired_difference(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    truth: pd.DataFrame,
    key: str = "Molecule_Name",
    n_repeats: int = BOOTSTRAP_SAMPLES,
) -> pd.DataFrame:
    """Compare two submissions on identical bootstrap samples.

    Because the resampling seed is fixed, both submissions are scored on exactly the
    same draws, so their difference can be taken *within* each sample. That paired
    difference is far more sensitive than comparing two independent standard
    deviations, and it is how the challenge's own head-to-head comparison works.

    A difference much smaller than the standalone standard deviation can still be
    real if it is consistent across samples, and the p-value is what says so.

    Args:
        predictions_a: The submission under test.
        predictions_b: The reference submission.
        truth: Ground truth with credible bounds.
        key: Compound identifier column.
        n_repeats: Bootstrap samples.

    Returns:
        Per endpoint and macro: the mean paired difference in ST-RAE, its standard
        deviation, a one-sided bootstrap p-value, and the number of samples in which
        ``a`` scores lower than ``b``.
    """
    per_sample_a = _bootstrap_all_endpoints(predictions_a, truth, key, n_repeats)
    per_sample_b = _bootstrap_all_endpoints(predictions_b, truth, key, n_repeats)
    rows = []
    for endpoint in per_sample_a:
        difference = per_sample_a[endpoint][PRIMARY_METRIC] - per_sample_b[endpoint][PRIMARY_METRIC]
        rows.append(
            {
                "endpoint": endpoint,
                "delta_mean": difference.mean(),
                "delta_std": difference.std(ddof=1),
                # One-sided bootstrap p-value: the share of samples in which A does
                # *not* beat B. The challenge reports a ``p_value`` column and
                # displays "won N/N bootstrap samples" for the same comparison.
                "p_value": float((difference >= 0).mean()),
                "n_wins": int((difference < 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("endpoint")
