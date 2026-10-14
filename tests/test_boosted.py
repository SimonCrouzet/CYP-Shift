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
    objective = soft_threshold_gradient(lower, upper, np.ones(2))
    gradient, _ = objective(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
    assert (gradient == 0).all()


def test_gradient_points_down_when_predicting_too_high():
    objective = soft_threshold_gradient(np.array([4.8]), np.array([5.2]), np.ones(1))
    gradient, _ = objective(np.array([5.0]), np.array([9.0]))
    assert gradient[0] == 1.0  # positive gradient means the step reduces the prediction


def test_gradient_points_up_when_predicting_too_low():
    objective = soft_threshold_gradient(np.array([4.8]), np.array([5.2]), np.ones(1))
    gradient, _ = objective(np.array([5.0]), np.array([1.0]))
    assert gradient[0] == -1.0


def test_hessian_is_constant_and_positive():
    """The true second derivative is zero, which LightGBM cannot use."""
    objective = soft_threshold_gradient(np.array([4.8, 4.8]), np.array([5.2, 5.2]), np.ones(2))
    _, hessian = objective(np.array([5.0, 5.0]), np.array([1.0, 9.0]))
    assert (hessian > 0).all()
    assert len(set(hessian)) == 1


def test_the_row_weight_scales_the_gradient_and_hessian():
    """LightGBM does not weight a custom objective, so the weight is applied here."""
    weight = np.array([0.0, 1.0, 10.0])
    objective = soft_threshold_gradient(np.zeros(3), np.ones(3), weight)
    gradient, hessian = objective(np.zeros(3), np.full(3, 5.0))  # all above the band
    assert gradient.tolist() == [0.0, 1.0, 10.0]
    assert hessian.tolist() == [0.0, 1.0, 10.0]


def test_sample_weights_change_the_soft_threshold_fit():
    """Regression test: uniform versus non-uniform weights must fit different models.

    LightGBM 4.7.0 ignores ``sample_weight`` for a custom objective, so without the
    in-objective weighting every source, screen and distance weight was a no-op.
    """
    observations = _observations()
    uniform = observations.copy()
    uniform["weight"] = 1.0
    varied = observations.copy()
    rng = np.random.default_rng(0)
    varied["weight"] = rng.uniform(0.1, 10.0, len(varied))
    a = _model(loss="soft_threshold").fit(uniform).predict(SMILES).mean
    b = _model(loss="soft_threshold").fit(varied).predict(SMILES).mean
    assert not np.allclose(a, b)


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


def _two_arm_observations():
    import numpy as np
    import pandas as pd

    smiles = [
        "CN(C)CCOC(c1ccccc1)c1ccccc1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1CCC(CC1)Oc1ccccc1Cl",
        "Cc1ccc(cc1)S(N)(=O)=O",
        "CCN(CC)CCNC(=O)c1ccc(N)cc1",
        "COc1ccc2nc(N)sc2c1",
        "OC(=O)c1ccccc1O",
        "CCCCn1cnc2c1c(=O)n(C)c(=O)n2C",
    ]
    rng = np.random.default_rng(0)
    rows = []
    for arm in ("direct_inhibition_drc", "tdi_drc"):
        for i, smi in enumerate(smiles):
            for target in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
                value = rng.uniform(4, 7)
                rows.append(
                    dict(
                        Molecule_Name=f"M{i}",
                        SMILES=smi,
                        target=target,
                        assay=arm,
                        value=value,
                        conf_low=value,
                        conf_high=value,
                        weight=1.0,
                        is_censored=False,
                    )
                )
    return pd.DataFrame(rows), smiles


def test_boosted_predicts_tdi_from_two_scalar_values():
    """The 'two values' path: a scalar model does the classification track by predicting
    the direct and preincubated pIC50 and comparing them, matching the curve model's
    output shape."""
    import numpy as np

    from cypshift.models.boosted import BoostedTrees

    observations, smiles = _two_arm_observations()
    model = BoostedTrees(n_bits=512, n_estimators=8, loss="squared")
    model.fit(observations)
    flags = model.predict_is_time_dependent(smiles[:3])
    assert flags.shape == (3, 4)  # one column per isoform, like the curve model
    assert flags.dtype == np.bool_


def test_a_clear_shift_is_called_time_dependent():
    """If the preincubated pIC50 is well above the direct one, the rule fires."""
    import numpy as np

    from cypshift.config import TDI_SHIFT_THRESHOLD
    from cypshift.models.boosted import BoostedTrees

    model = BoostedTrees()
    # stub the per-assay prediction: a big direct->preincubated shift on every isoform
    model._predict_pic50_at = lambda smiles, enzyme, assay: (
        np.full(len(smiles), 6.0)
        if assay == "tdi_drc"
        else np.full(len(smiles), 6.0 - 2 * TDI_SHIFT_THRESHOLD)
    )
    flags = model.predict_is_time_dependent(["CCO", "CCN"])
    assert flags.all()
