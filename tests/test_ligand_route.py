"""The ligand-only baseline route."""

import numpy as np
import pytest

from cypshift.models.base import PredictiveDistribution
from cypshift.models.ligand import LigandOnlyRoute

ENDPOINTS = ("CYP1A2_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")
SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "CC(=O)N({})c1ccccc1"]
SUBSTITUENTS = ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC", "OC", "C(F)(F)F"]
SMILES = [
    scaffold.format(substituent) for scaffold in SCAFFOLDS for substituent in SUBSTITUENTS
] * 2


def _targets(n, seed=0, missing_in_second=0):
    rng = np.random.default_rng(seed)
    targets = rng.uniform(3.5, 7.0, (n, len(ENDPOINTS)))
    if missing_in_second:
        targets[:missing_in_second, 1] = np.nan
    return targets


def _route(**kwargs):
    return LigandOnlyRoute(endpoints=ENDPOINTS, n_members=3, n_estimators=25, n_bits=256, **kwargs)


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        _route().predict(SMILES[:3])


def test_wrong_target_shape_is_rejected():
    with pytest.raises(ValueError, match="targets must be"):
        _route().fit(SMILES, np.zeros((len(SMILES), 3)))


def test_an_endpoint_with_too_few_labels_is_rejected():
    targets = _targets(len(SMILES))
    targets[5:, 1] = np.nan
    with pytest.raises(ValueError, match="only 5 labels"):
        _route().fit(SMILES, targets)


def test_fit_then_predict_returns_a_distribution_of_the_right_shape():
    route = _route().fit(SMILES, _targets(len(SMILES)))
    prediction = route.predict(SMILES[:7])
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.samples.shape == (3, 7, len(ENDPOINTS))
    assert prediction.endpoints == ENDPOINTS
    assert np.isfinite(prediction.samples).all()


def test_unmeasured_compounds_are_skipped_for_that_endpoint_only():
    route = _route().fit(SMILES, _targets(len(SMILES), missing_in_second=10))
    assert route.predict(SMILES[:4]).samples.shape == (3, 4, len(ENDPOINTS))


def test_members_disagree_so_the_ensemble_carries_a_spread():
    route = _route().fit(SMILES, _targets(len(SMILES)))
    prediction = route.predict(SMILES)
    assert prediction.epistemic_variance.mean() > 0


def test_predictions_land_in_the_range_of_the_training_labels():
    targets = _targets(len(SMILES))
    route = _route().fit(SMILES, targets)
    prediction = route.predict(SMILES).mean
    assert prediction.min() >= np.nanmin(targets) - 1.0
    assert prediction.max() <= np.nanmax(targets) + 1.0


def test_the_route_satisfies_the_declared_interface():
    route = _route()
    assert route.name == "ligand"
    assert hasattr(route, "fit") and hasattr(route, "predict")
