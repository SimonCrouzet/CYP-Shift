"""Tests for the time-dependent inhibition track's scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score

from cypshift.config import CLASSIFICATION_ENDPOINTS, MACRO_LABEL
from cypshift.eval.classification import (
    CLASSIFICATION_METRICS,
    PRIMARY_CLASSIFICATION_METRIC,
    score_classification,
    score_classification_endpoint,
)

ENDPOINTS = tuple(CLASSIFICATION_ENDPOINTS)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return pd.DataFrame({e: rng.random(200) < 0.22 for e in ENDPOINTS}).astype("boolean")


def test_the_official_metric_list_is_mirrored():
    """Taken from the organisers' evaluation/config.py, not chosen here."""
    assert [name for name, _ in CLASSIFICATION_METRICS] == [
        "MCC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]
    assert PRIMARY_CLASSIFICATION_METRIC == "MCC"


def test_only_two_endpoints_are_classified():
    assert set(ENDPOINTS) == {"CYP2D6_is_TDI", "CYP3A4_is_TDI"}


def test_endpoint_scores_match_sklearn_directly():
    truth = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    scored = score_classification_endpoint(truth, predicted)
    assert scored["MCC"] == pytest.approx(matthews_corrcoef(truth, predicted))
    assert scored["F1"] == pytest.approx(f1_score(truth, predicted, zero_division=0))
    assert scored["Precision"] == pytest.approx(precision_score(truth, predicted, zero_division=0))


def test_a_resample_with_no_positive_prediction_does_not_raise():
    """zero_division=0 is the organisers' setting, and it matters: TDI labels are
    imbalanced enough that a bootstrap sample can contain no positive prediction."""
    scored = score_classification_endpoint(np.array([1, 0, 0]), np.array([0, 0, 0]))
    assert scored["Precision"] == 0.0
    assert np.isfinite(list(scored.values())).all()


def test_perfect_prediction_scores_one(labels):
    scores = score_classification(labels.astype(bool), labels, n_repeats=20)
    assert scores.loc[MACRO_LABEL, "MCC_mean"] == pytest.approx(1.0)


def test_always_negative_scores_zero_mcc_despite_high_accuracy(labels):
    """The reason the track ranks by MCC. Answering "no" to everything is right about
    four times in five and has learned nothing."""
    negative = pd.DataFrame({e: [False] * len(labels) for e in ENDPOINTS})
    scores = score_classification(negative, labels, n_repeats=20)
    assert scores.loc[MACRO_LABEL, "MCC_mean"] == pytest.approx(0.0)
    assert scores.loc[MACRO_LABEL, "Accuracy_mean"] > 0.7


def test_the_macro_row_is_present_and_last(labels):
    scores = score_classification(labels.astype(bool), labels, n_repeats=10)
    assert scores.index[-1] == MACRO_LABEL


def test_the_macro_row_averages_the_endpoints(labels):
    scores = score_classification(labels.astype(bool), labels, n_repeats=10)
    endpoints = [e for e in ENDPOINTS if e in scores.index]
    assert scores.loc[MACRO_LABEL, "MCC_mean"] == pytest.approx(
        scores.loc[endpoints, "MCC_mean"].mean(), abs=1e-9
    )


def test_rows_without_a_label_are_dropped_per_endpoint(labels):
    """The TDI file covers different compounds per enzyme: 1,497 for CYP2D6 against
    3,584 for CYP3A4."""
    partial = labels.copy()
    partial.loc[:100, ENDPOINTS[0]] = pd.NA
    scores = score_classification(labels.astype(bool), partial, n_repeats=10)
    assert scores.loc[ENDPOINTS[0], "MCC_mean"] == pytest.approx(1.0)


def test_scoring_is_reproducible(labels):
    predicted = labels.astype(bool).copy()
    predicted.iloc[:20] = ~predicted.iloc[:20]
    first = score_classification(predicted, labels, n_repeats=25)
    second = score_classification(predicted, labels, n_repeats=25)
    pd.testing.assert_frame_equal(first, second)


def test_no_shared_endpoint_is_refused(labels):
    with pytest.raises(ValueError, match="no classification endpoint"):
        score_classification(pd.DataFrame({"other": [True]}), labels, n_repeats=5)
