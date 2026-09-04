"""The architecturally diverse ensemble."""

import numpy as np
import pytest

from cypshift.models.base import PredictiveDistribution
from cypshift.models.ensemble import DEFAULT_MEMBERS, MemberSpec, PerTargetEnsemble
from cypshift.models.forest import PerTargetForest

ENDPOINTS = ("CYP1A2_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")
SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "CC(=O)N({})c1ccccc1", "Cn1cnc({})c1"]
SUBSTITUENTS = ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC", "OC", "C(F)(F)F"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in SUBSTITUENTS] * 2


def _targets(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(3.5, 7.0, (n, len(ENDPOINTS)))


def _fast_members():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor

    return (
        MemberSpec(
            "random_forest_all", lambda s: RandomForestRegressor(n_estimators=20, random_state=s)
        ),
        MemberSpec(
            "random_forest_descriptors",
            lambda s: RandomForestRegressor(n_estimators=20, random_state=s),
            fingerprint=False,
        ),
        MemberSpec(
            "nearest_neighbours_fingerprint",
            lambda s: KNeighborsRegressor(n_neighbors=3, metric="jaccard"),
            physchem=False,
            mechanism=False,
        ),
    )


def _route():
    return PerTargetEnsemble(endpoints=ENDPOINTS, members=_fast_members(), n_bits=256)


def test_default_members_span_more_than_one_feature_view():
    assert len({(m.fingerprint, m.physchem, m.mechanism) for m in DEFAULT_MEMBERS}) > 1


def test_default_members_span_more_than_one_estimator_family():
    families = {type(spec.build(0)).__name__ for spec in DEFAULT_MEMBERS}
    assert len(families) >= 3


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        _route().predict(SMILES[:3])


def test_wrong_target_shape_is_rejected():
    with pytest.raises(ValueError, match="targets must be"):
        _route().fit(SMILES, np.zeros((len(SMILES), 3)))


def test_fit_then_predict_returns_one_sample_per_member():
    model = _route().fit(SMILES, _targets(len(SMILES)))
    prediction = model.predict(SMILES[:6])
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.samples.shape == (3, 6, len(ENDPOINTS))
    assert np.isfinite(prediction.samples).all()


def test_diverse_members_disagree_more_than_reseeded_ones():
    """The point of the model: real disagreement, not seed noise."""
    targets = _targets(len(SMILES))
    diverse = _route().fit(SMILES, targets).predict(SMILES)
    seeded = (
        PerTargetForest(endpoints=ENDPOINTS, n_members=3, n_estimators=20, n_bits=256)
        .fit(SMILES, targets)
        .predict(SMILES)
    )
    assert diverse.epistemic_variance.mean() > seeded.epistemic_variance.mean()


def test_an_endpoint_with_too_few_labels_is_rejected():
    targets = _targets(len(SMILES))
    targets[5:, 1] = np.nan
    with pytest.raises(ValueError, match="only 5 labels"):
        _route().fit(SMILES, targets)


def test_the_route_satisfies_the_declared_interface():
    model = _route()
    assert model.name == "ligand_diverse"
    assert hasattr(model, "fit") and hasattr(model, "predict")
