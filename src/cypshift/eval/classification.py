"""Bootstrapped scoring of the time-dependent inhibition track.

The challenge has two tracks and this is the second. It asks a yes/no question for two
enzymes only -- ``CYP2D6_is_TDI`` and ``CYP3A4_is_TDI`` -- and ranks by Matthews
correlation, not by accuracy. That choice matters: TDI labels are heavily imbalanced,
so a model answering "no" to everything scores well on accuracy and zero on MCC, which
is the value MCC gives a coin flip.

Metrics and their degenerate-case handling are taken from the organisers'
``evaluation/config.py`` rather than chosen here:

============  ==========================================================
MCC           ``matthews_corrcoef``; already returns 0.0 rather than NaN
Accuracy      ``accuracy_score``
Precision     ``precision_score(zero_division=0)``
Recall        ``recall_score(zero_division=0)``
F1            ``f1_score(zero_division=0)``
============  ==========================================================

``zero_division=0`` is theirs, and it matters here: a bootstrap resample of an
imbalanced label can contain no positive prediction at all, and the default would raise
or warn instead of scoring it.

Resampling follows :mod:`cypshift.eval.scoring` exactly -- same seed, each endpoint
resampled over its own row count, and the macro average taken inside each sample so it
carries its own bootstrap distribution.
"""

import logging
from functools import partial

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from cypshift.config import CLASSIFICATION_ENDPOINTS, MACRO_LABEL
from cypshift.eval.scoring import bootstrap_indices

logger = logging.getLogger(__name__)

#: Mirrors the organisers' ``CLASSIFICATION_METRICS``, in their order.
CLASSIFICATION_METRICS = (
    ("MCC", matthews_corrcoef),
    ("Accuracy", accuracy_score),
    ("Precision", partial(precision_score, zero_division=0)),
    ("Recall", partial(recall_score, zero_division=0)),
    ("F1", partial(f1_score, zero_division=0)),
)

#: What the classification leaderboard sorts on. Higher is better, 0.0 is chance.
PRIMARY_CLASSIFICATION_METRIC = "MCC"


def score_classification_endpoint(
    y_true: NDArray[np.bool_], y_predicted: NDArray[np.bool_]
) -> dict[str, float]:
    """Every classification metric for one endpoint, on one sample."""
    truth, predicted = np.asarray(y_true).astype(int), np.asarray(y_predicted).astype(int)
    return {name: float(metric(truth, predicted)) for name, metric in CLASSIFICATION_METRICS}


def _bootstrap_endpoints(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    endpoints: tuple[str, ...],
    n_repeats: int,
) -> dict[str, list[dict[str, float]]]:
    """Per-endpoint metric values across bootstrap samples, plus the macro row."""
    per_endpoint: dict[str, list[dict[str, float]]] = {e: [] for e in endpoints}
    per_endpoint[MACRO_LABEL] = []

    usable = {}
    for endpoint in endpoints:
        if endpoint not in truth.columns or endpoint not in submission.columns:
            logger.info("skipping %s: absent from the submission or the labels", endpoint)
            continue
        mask = truth[endpoint].notna()
        usable[endpoint] = (
            truth.loc[mask, endpoint].to_numpy(),
            submission.loc[mask, endpoint].to_numpy(),
            bootstrap_indices(int(mask.sum()), n_repeats),
        )
    if not usable:
        raise ValueError("no classification endpoint is present in both tables")

    for sample in range(n_repeats):
        collected = []
        for endpoint, (y_true, y_pred, samples) in usable.items():
            index = samples[sample]
            scored = score_classification_endpoint(y_true[index], y_pred[index])
            per_endpoint[endpoint].append(scored)
            collected.append(scored)
        per_endpoint[MACRO_LABEL].append(
            {
                name: float(np.mean([c[name] for c in collected]))
                for name, _ in CLASSIFICATION_METRICS
            }
        )
    return {k: v for k, v in per_endpoint.items() if v}


def score_classification(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    endpoints: tuple[str, ...] = CLASSIFICATION_ENDPOINTS,
    n_repeats: int = 1000,
) -> pd.DataFrame:
    """Score the TDI track, one row per endpoint plus the macro average.

    Args:
        submission: Boolean or 0/1 predictions, one column per endpoint.
        truth: The same columns, holding labels. Rows missing a label are dropped for
            that endpoint only.
        endpoints: Which endpoints to score.
        n_repeats: Bootstrap samples.

    Returns:
        Mean and standard deviation of each metric, indexed by endpoint with
        :data:`cypshift.config.MACRO_LABEL` last.
    """
    collected = _bootstrap_endpoints(submission, truth, endpoints, n_repeats)
    rows = {}
    for endpoint, samples in collected.items():
        frame = pd.DataFrame(samples)
        rows[endpoint] = {
            **{f"{c}_mean": frame[c].mean() for c in frame.columns},
            **{f"{c}_std": frame[c].std() for c in frame.columns},
        }
    table = pd.DataFrame(rows).T
    table.index.name = "endpoint"
    ordered = [e for e in endpoints if e in table.index] + [MACRO_LABEL]
    return table.loc[[e for e in ordered if e in table.index]]
