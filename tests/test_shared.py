"""The structure-free model conditioned on target and assay."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import build_observations
from cypshift.models.base import PredictiveDistribution
from cypshift.models.ensemble import MemberSpec
from cypshift.models.shared import SharedEnsemble

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SUBSTITUENTS = ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in SUBSTITUENTS]


def _inhibition(seed=0):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:05d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        frame[f"{isoform}_pIC50_direct_inhibition"] = rng.uniform(3.0, 7.0, len(SMILES))
    return pd.DataFrame(frame)


def _screen(seed=1):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"SCRN-{i:05d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        frame[f"{isoform}_pIC50_direct_inhibition"] = rng.uniform(2.0, 4.0, len(SMILES))
        frame[f"{isoform}_is_censored"] = False
    return pd.DataFrame(frame)


def _members():
    return (
        MemberSpec(
            "random_forest_all", lambda s: RandomForestRegressor(n_estimators=20, random_state=s)
        ),
        MemberSpec(
            "random_forest_descriptors",
            lambda s: RandomForestRegressor(n_estimators=20, random_state=s),
            fingerprint=False,
        ),
    )


def _route():
    return SharedEnsemble(members=_members(), n_bits=256)


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        _route().predict(SMILES[:3])


def test_fitting_on_an_empty_table_is_rejected():
    empty = build_observations(_inhibition()).iloc[:0]
    with pytest.raises(ValueError, match="no finite measurements"):
        _route().fit(empty)


def test_one_model_serves_every_endpoint():
    model = _route().fit(build_observations(_inhibition()))
    prediction = model.predict(SMILES[:5])
    assert isinstance(prediction, PredictiveDistribution)
    assert prediction.samples.shape == (2, 5, len(REGRESSION_ENDPOINTS))
    assert np.isfinite(prediction.samples).all()


def test_the_same_molecule_gets_different_predictions_per_isoform():
    """Only the one_hot block distinguishes them, so it must be doing work."""
    model = _route().fit(build_observations(_inhibition()))
    prediction = model.predict(SMILES[:8]).mean
    spread_across_isoforms = prediction.std(axis=1)
    assert (spread_across_isoforms > 0).all()


def test_a_screen_source_can_be_trained_on_without_a_separate_model():
    observations = build_observations(_inhibition(), screen_labels=_screen())
    model = _route().fit(observations)
    assert model.predict(SMILES[:4]).samples.shape == (2, 4, len(REGRESSION_ENDPOINTS))


def test_adding_a_source_changes_what_the_model_learned():
    curves_only = _route().fit(build_observations(_inhibition())).predict(SMILES).mean
    with_screen = (
        _route()
        .fit(build_observations(_inhibition(), screen_labels=_screen()))
        .predict(SMILES)
        .mean
    )
    assert not np.allclose(curves_only, with_screen)


def test_prediction_is_made_at_the_scored_assay():
    from cypshift.models.shared import SCORED_ASSAY

    assert SCORED_ASSAY == "direct_inhibition_drc"


def test_the_route_satisfies_the_declared_interface():
    model = _route()
    assert model.name == "structure_free"
    assert hasattr(model, "fit") and hasattr(model, "predict")
