"""The analogue-based local correction (cypshift.decision.analogue)."""

import numpy as np
import pytest

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.decision.analogue import AnalogueCorrector, _interval_point

ENDPOINT = REGRESSION_ENDPOINTS[0]


def _single_endpoint(values: dict[str, float]) -> np.ndarray:
    """A (n, n_endpoints) array with ``values`` on the first endpoint, NaN elsewhere."""
    out = np.full((len(values), len(REGRESSION_ENDPOINTS)), np.nan)
    out[:, 0] = list(values.values())
    return out


def test_interval_point_sits_inside_a_lone_interval():
    """One interval: any point inside is optimal, and the minimiser is one of its bounds."""
    point = _interval_point(np.array([2.0]), np.array([3.0]), np.array([1.0]))
    assert 2.0 <= point <= 3.0


def test_interval_point_is_pulled_toward_the_heavier_cluster():
    """Two disjoint intervals: the heavier one wins, and the optimum is its near edge."""
    low = np.array([0.0, 5.0])
    high = np.array([1.0, 6.0])
    weights = np.array([3.0, 1.0])  # the [0, 1] interval carries three times the weight
    assert _interval_point(low, high, weights) == pytest.approx(1.0)


def _fit_corrector(estimator: str, **kwargs) -> AnalogueCorrector:
    # Two close analogues (ethanol/propanol) the model under-predicts by ~1 log unit, and a
    # distant aromatic it gets right. The references' truth, bounds and model prediction.
    smiles = ["CCO", "CCCO", "c1ccccc1"]
    truth = _single_endpoint({"a": 5.0, "b": 5.2, "c": 8.0})
    model_pred = _single_endpoint({"a": 4.0, "b": 4.1, "c": 7.95})
    lower, upper = truth - 0.3, truth + 0.3
    return AnalogueCorrector(estimator=estimator, **kwargs).fit(
        smiles, truth, lower, upper, model_pred
    )


def test_correct_preserves_shape():
    corrector = _fit_corrector("residual")
    out = corrector.correct(["CCO", "c1ccccc1"], np.zeros((2, len(REGRESSION_ENDPOINTS))))
    assert out.shape == (2, len(REGRESSION_ENDPOINTS))


def test_a_compound_with_no_analogue_is_left_unchanged():
    """A query far from every reference has no eligible neighbour, so the global value stands."""
    corrector = _fit_corrector("residual", min_similarity=0.4)
    distant = "C1CC2CCC1C2"  # a cage, unlike any reference
    prediction = np.full((1, len(REGRESSION_ENDPOINTS)), 3.0)
    out = corrector.correct([distant], prediction)
    assert out[0, 0] == pytest.approx(3.0)


def test_residual_estimator_corrects_a_local_underprediction_upward():
    """A query on top of references the model under-predicts is pulled up toward their level."""
    corrector = _fit_corrector("residual", min_similarity=0.3, shrinkage=0.5)
    prediction = _single_endpoint({"q": 4.0})  # the model's (low) prediction for an ethanol-like q
    out = corrector.correct(["CCO"], prediction)
    assert out[0, 0] > 4.0  # the +1 residual of its neighbours pulls it up


def test_shrinkage_toward_one_trusts_the_neighbours_more():
    """A smaller shrinkage pseudo-count moves the prediction further toward the local estimate."""
    weak = _fit_corrector("residual", min_similarity=0.3, shrinkage=5.0)
    strong = _fit_corrector("residual", min_similarity=0.3, shrinkage=0.2)
    prediction = _single_endpoint({"q": 4.0})
    moved_weak = weak.correct(["CCO"], prediction)[0, 0] - 4.0
    moved_strong = strong.correct(["CCO"], prediction)[0, 0] - 4.0
    assert moved_strong > moved_weak > 0.0


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="estimator"):
        AnalogueCorrector(estimator="wishing")
