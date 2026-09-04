"""Bootstrapping and macro-averaging must match the leaderboard's conventions."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.eval.scoring import bootstrap_indices, score_submission

RNG = np.random.default_rng(11)


def _frames(n=60, missing_in_first_endpoint=0):
    names = [f"OCNT-{i:05d}" for i in range(n)]
    truth = {"Molecule_Name": names}
    preds = {"Molecule_Name": names}
    for endpoint in REGRESSION_ENDPOINTS:
        y = RNG.uniform(3, 7, n)
        truth[endpoint] = y
        truth[f"{endpoint}_conf_low"] = y - 0.2
        truth[f"{endpoint}_conf_high"] = y + 0.2
        preds[endpoint] = y + RNG.normal(0, 0.5, n)
    if missing_in_first_endpoint:
        truth[REGRESSION_ENDPOINTS[0]][:missing_in_first_endpoint] = np.nan
    return pd.DataFrame(preds), pd.DataFrame(truth)


def test_bootstrap_indices_are_seeded_and_reproducible():
    first = bootstrap_indices(50, 10)
    second = bootstrap_indices(50, 10)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (10, 50)
    assert first.min() >= 0 and first.max() < 50


def test_scoring_produces_a_row_per_endpoint_plus_the_macro():
    predictions, truth = _frames()
    scores = score_submission(predictions, truth, n_repeats=25)
    assert list(scores.index) == [*REGRESSION_ENDPOINTS, MACRO_LABEL]
    for column in ("ST-RAE_mean", "ST-RAE_std", "MAE_mean", "R2_mean"):
        assert column in scores.columns


def test_macro_is_the_plain_mean_of_the_endpoint_means():
    predictions, truth = _frames()
    scores = score_submission(predictions, truth, n_repeats=25)
    endpoint_means = scores.loc[list(REGRESSION_ENDPOINTS), "ST-RAE_mean"]
    assert scores.loc[MACRO_LABEL, "ST-RAE_mean"] == pytest.approx(endpoint_means.mean())


def test_missing_truth_drops_the_compound_for_that_endpoint_only():
    predictions, truth = _frames(missing_in_first_endpoint=10)
    scores = score_submission(predictions, truth, n_repeats=10)
    assert np.isfinite(scores.loc[MACRO_LABEL, "ST-RAE_mean"])


def test_a_missing_prediction_where_truth_exists_is_rejected():
    predictions, truth = _frames()
    predictions.loc[0, REGRESSION_ENDPOINTS[0]] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        score_submission(predictions, truth, n_repeats=5)


def test_the_mean_predictor_scores_about_one_on_every_endpoint():
    predictions, truth = _frames()
    for endpoint in REGRESSION_ENDPOINTS:
        predictions[endpoint] = truth[endpoint].mean()
    scores = score_submission(predictions, truth, n_repeats=50)
    # Not exactly 1.0: each resample has its own mean, which the constant does not track.
    assert scores.loc[MACRO_LABEL, "ST-RAE_mean"] == pytest.approx(1.0, abs=0.05)


def test_a_better_model_scores_lower():
    predictions, truth = _frames()
    good = score_submission(predictions, truth, n_repeats=25).loc[MACRO_LABEL, "ST-RAE_mean"]
    for endpoint in REGRESSION_ENDPOINTS:
        predictions[endpoint] = truth[endpoint] + RNG.normal(0, 2.0, len(truth))
    bad = score_submission(predictions, truth, n_repeats=25).loc[MACRO_LABEL, "ST-RAE_mean"]
    assert good < bad
