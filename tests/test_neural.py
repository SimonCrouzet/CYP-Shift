"""The soft-threshold objective, and the model trained on it."""

import numpy as np
import pandas as pd
import pytest
import torch

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import build_observations
from cypshift.models.base import PredictiveDistribution
from cypshift.models.neural import NeuralNetwork, soft_threshold_loss

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]]


def _tensor(*values):
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def test_a_prediction_inside_the_band_costs_nothing():
    pred, low, high, w = _tensor([5.0, 6.0], [4.8, 5.8], [5.2, 6.2], [1.0, 1.0])
    assert float(soft_threshold_loss(pred, low, high, w)) == 0.0


def test_a_prediction_on_a_bound_costs_nothing():
    pred, low, high, w = _tensor([5.2], [4.8], [5.2], [1.0])
    assert float(soft_threshold_loss(pred, low, high, w)) == pytest.approx(0.0)


def test_cost_is_distance_to_the_nearest_bound():
    pred, low, high, w = _tensor([6.2], [4.8], [5.2], [1.0])
    assert float(soft_threshold_loss(pred, low, high, w)) == pytest.approx(1.0)


def test_gradient_vanishes_inside_the_band():
    """The reason to train on this at all: no capacity spent where nothing is charged."""
    pred = torch.tensor([5.0], dtype=torch.float32, requires_grad=True)
    low, high, w = _tensor([4.8], [5.2], [1.0])
    soft_threshold_loss(pred, low, high, w).backward()
    assert float(pred.grad) == 0.0


def test_gradient_is_present_outside_the_band():
    pred = torch.tensor([6.0], dtype=torch.float32, requires_grad=True)
    low, high, w = _tensor([4.8], [5.2], [1.0])
    soft_threshold_loss(pred, low, high, w).backward()
    assert float(pred.grad) > 0.0


def test_a_censored_label_charges_nothing_below_its_bound():
    """A one-sided interval is what a screen reading outside the calibrated range supports."""
    pred, low, high, w = _tensor([1.0, 3.0], [-1e4, -1e4], [3.86, 3.86], [1.0, 1.0])
    assert float(soft_threshold_loss(pred, low, high, w)) == 0.0


def test_a_censored_label_still_charges_above_its_bound():
    pred, low, high, w = _tensor([5.86], [-1e4], [3.86], [1.0])
    assert float(soft_threshold_loss(pred, low, high, w)) == pytest.approx(2.0)


def test_weights_scale_a_row_contribution():
    pred, low, high = _tensor([6.2, 6.2], [5.2, 5.2], [5.2, 5.2])
    trusted = soft_threshold_loss(pred, low, high, torch.tensor([1.0, 1.0]))
    halved = soft_threshold_loss(pred[:1], low[:1], high[:1], torch.tensor([0.3]))
    assert float(trusted) == pytest.approx(float(halved))  # weighted mean, not sum


def _observations(seed=0):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        y = rng.uniform(3.0, 7.0, len(SMILES))
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    return build_observations(pd.DataFrame(frame))


def _route(**kwargs):
    return NeuralNetwork(n_bits=128, hidden=32, epochs=5, n_forward_passes=4, **kwargs)


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        _route().predict(SMILES[:3])


def test_fitting_on_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="no finite measurements"):
        _route().fit(_observations().iloc[:0])


def test_fit_then_predict_returns_a_distribution():
    model = _route().fit(_observations())
    prediction = model.predict(SMILES[:5])
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.samples.shape == (4, 5, len(REGRESSION_ENDPOINTS))
    assert np.isfinite(prediction.samples).all()


def test_dropout_at_prediction_time_produces_a_spread():
    model = _route(dropout=0.3).fit(_observations())
    assert model.predict(SMILES).epistemic_variance.mean() > 0


def test_the_two_objectives_learn_different_models():
    observations = _observations()
    dead = _route(loss="soft_threshold").fit(observations).predict(SMILES).mean
    squared = _route(loss="squared").fit(observations).predict(SMILES).mean
    assert not np.allclose(dead, squared)


def test_every_declared_objective_trains():
    from cypshift.models.neural import OBJECTIVES

    observations = _observations()
    for objective in OBJECTIVES:
        model = _route(loss=objective).fit(observations)
        assert np.isfinite(model.predict(SMILES[:3]).mean).all()


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        _route(loss="not_an_objective").fit(_observations())


def test_absolute_and_squared_objectives_differ():
    observations = _observations()
    absolute = _route(loss="absolute").fit(observations).predict(SMILES).mean
    squared = _route(loss="squared").fit(observations).predict(SMILES).mean
    assert not np.allclose(absolute, squared)
