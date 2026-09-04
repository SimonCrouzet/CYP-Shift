"""Boosted trees with a custom soft-threshold objective."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import build_observations
from cypshift.models.base import PredictiveDistribution
from cypshift.models.boosted import BoostedTrees, soft_threshold_gradient

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]]


def test_gradient_is_zero_inside_the_credible_interval():
    lower, upper = np.array([4.8, 5.8]), np.array([5.2, 6.2])
    objective = soft_threshold_gradient(lower, upper)
    gradient, _ = objective(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
    assert (gradient == 0).all()


def test_gradient_points_down_when_predicting_too_high():
    objective = soft_threshold_gradient(np.array([4.8]), np.array([5.2]))
    gradient, _ = objective(np.array([5.0]), np.array([9.0]))
    assert gradient[0] == 1.0  # positive gradient means the step reduces the prediction


def test_gradient_points_up_when_predicting_too_low():
    objective = soft_threshold_gradient(np.array([4.8]), np.array([5.2]))
    gradient, _ = objective(np.array([5.0]), np.array([1.0]))
    assert gradient[0] == -1.0


def test_hessian_is_constant_and_positive():
    """The true second derivative is zero, which LightGBM cannot use."""
    objective = soft_threshold_gradient(np.array([4.8, 4.8]), np.array([5.2, 5.2]))
    _, hessian = objective(np.array([5.0, 5.0]), np.array([1.0, 9.0]))
    assert (hessian > 0).all()
    assert len(set(hessian)) == 1


def _observations(seed=0):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        y = rng.uniform(3.0, 7.0, len(SMILES))
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    return build_observations(pd.DataFrame(frame))


def _model(**kwargs):
    return BoostedTrees(n_bits=128, n_estimators=30, n_members=2, num_leaves=7, **kwargs)


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="model is not fitted"):
        _model().predict(SMILES[:3])


def test_fitting_on_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="no finite measurements"):
        _model().fit(_observations().iloc[:0])


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        _model(loss="not_an_objective").fit(_observations())


def test_fit_then_predict_returns_a_distribution():
    model = _model().fit(_observations())
    prediction = model.predict(SMILES[:5])
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.samples.shape == (2, 5, len(REGRESSION_ENDPOINTS))
    assert np.isfinite(prediction.samples).all()


def test_predictions_are_on_the_pic50_scale_not_the_residual_scale():
    """A custom objective boosts from zero, so the offset must be added back."""
    observations = _observations()
    prediction = _model().fit(observations).predict(SMILES).mean
    assert 2.0 < prediction.mean() < 8.0


def test_every_objective_trains():
    observations = _observations()
    for objective in ("squared", "absolute", "soft_threshold"):
        model = _model(loss=objective).fit(observations)
        assert np.isfinite(model.predict(SMILES[:3]).mean).all()


def test_the_objectives_learn_different_models():
    observations = _observations()
    soft = _model(loss="soft_threshold").fit(observations).predict(SMILES).mean
    squared = _model(loss="squared").fit(observations).predict(SMILES).mean
    assert not np.allclose(soft, squared)


def test_the_model_satisfies_the_declared_interface():
    model = _model()
    assert model.name == "boosted_trees"
    assert hasattr(model, "fit") and hasattr(model, "predict")
