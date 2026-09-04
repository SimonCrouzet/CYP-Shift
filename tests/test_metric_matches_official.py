"""Our metric must agree numerically with the challenge's own scorer.

Looking similar is not enough. The reference in ``tests/reference/`` is a verbatim
copy of the function the organisers publish, so this asserts equality on random cases
and on the real released data. If the challenge updates its scorer, replacing that
file makes this test report exactly what moved.
"""

import numpy as np
import pytest

from cypshift.eval.metrics import soft_threshold_rae
from tests.reference.official_metric import rae_soft_threshold_absolute_error as official

RNG = np.random.default_rng(0)


def _case(n=200):
    """Truth with realistic activity-dependent bands: wide below the assay floor."""
    y = RNG.uniform(2, 8, n)
    half = np.where(y < 4, RNG.uniform(0.6, 1.4, n), RNG.uniform(0.1, 0.3, n))
    return y, y - half, y + half


@pytest.mark.parametrize("trial", range(25))
def test_agrees_with_the_official_scorer_on_random_cases(trial):
    y, lower, upper = _case()
    prediction = y + RNG.normal(0, RNG.uniform(0.1, 2.0), len(y))
    assert soft_threshold_rae(y, prediction, lower, upper) == pytest.approx(
        official(y, prediction, y_true_upper=upper, y_true_lower=lower), rel=1e-12
    )


def test_agrees_when_every_prediction_is_inside_its_band():
    y, lower, upper = _case()
    assert soft_threshold_rae(y, y, lower, upper) == pytest.approx(
        official(y, y, y_true_upper=upper, y_true_lower=lower)
    )


def test_agrees_on_a_wildly_wrong_submission():
    y, lower, upper = _case()
    prediction = np.full_like(y, 12.0)
    assert soft_threshold_rae(y, prediction, lower, upper) == pytest.approx(
        official(y, prediction, y_true_upper=upper, y_true_lower=lower), rel=1e-12
    )


def test_agrees_on_the_released_training_data():
    """Synthetic agreement is weaker evidence than agreement on the real bounds."""
    pandas = pytest.importorskip("pandas")
    path = "data/raw/cyp-challenge-TRAIN_inhibition.csv"
    try:
        frame = pandas.read_csv(path)
    except FileNotFoundError:
        pytest.skip("challenge data not downloaded; run `cypshift fetch`")

    for endpoint in [c for c in frame.columns if c.endswith("_pIC50_direct_inhibition")]:
        measured = frame[endpoint].notna()
        y = frame.loc[measured, endpoint].to_numpy(float)
        lower = frame.loc[measured, f"{endpoint}_conf_low"].to_numpy(float)
        upper = frame.loc[measured, f"{endpoint}_conf_high"].to_numpy(float)
        prediction = y + RNG.normal(0, 0.5, len(y))
        assert soft_threshold_rae(y, prediction, lower, upper) == pytest.approx(
            official(y, prediction, y_true_upper=upper, y_true_lower=lower), rel=1e-12
        )
