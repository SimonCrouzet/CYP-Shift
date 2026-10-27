"""The tabular foundation model's feature path: the shared set, compressed to fit TabPFN.

These tests cover the modular contract -- that the TFM assembles the same configured feature
set as the other models and only adapts its dimensionality -- without needing TabPFN itself
(the per-endpoint fit, which downloads weights and wants a token, is not exercised here).
"""

import numpy as np
import pandas as pd

from cypshift.models.tfm import TabularFoundationModel

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN(CC)CC",
    "c1ccncc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "CCCCCCCC",
    "c1ccc2ccccc2c1",
    "OCC1OC(O)C(O)C(O)C1O",
    "CC(N)C(=O)O",
    "c1ccc(cc1)O",
    "ClCCl",
    "FC(F)F",
    "CCOCC",
    "CC#N",
    "C=CC=C",
    "CC(C)=O",
]


def test_raw_features_include_the_fingerprint():
    """The TFM assembles the fingerprint like every other model -- not a dense-only set."""
    model = TabularFoundationModel(n_bits=256, physchem=True, mechanism=True)
    width = model._raw_features(SMILES).shape[1]
    assert width >= 256  # at least the 256 fingerprint bits, plus the dense blocks


def test_pca_adaptation_compresses_to_the_configured_width():
    """impute -> scale -> PCA reduces the shared feature set to pca_components."""
    model = TabularFoundationModel(n_bits=256, pca_components=8)
    model._fit_reducer(model._raw_features(SMILES))
    reduced = model._features(SMILES)
    assert reduced.shape == (len(SMILES), 8)


def test_pca_width_capped_by_available_rank():
    """When PCA applies, its width is capped by the data's rank (n_samples), never an error."""
    # pca_components below the feature count (273) triggers PCA; it is then capped by samples.
    model = TabularFoundationModel(n_bits=256, pca_components=100)
    model._fit_reducer(model._raw_features(SMILES))
    assert model._features(SMILES).shape[1] <= len(SMILES)


def test_no_pca_when_set_already_fits_under_ceiling():
    """A feature set already within pca_components is passed through unchanged -- no PCA."""
    model = TabularFoundationModel(n_bits=256, pca_components=10_000)
    model._fit_reducer(model._raw_features(SMILES))
    assert model._features(SMILES).shape[1] == model._raw_features(SMILES).shape[1]


def test_dense_only_feature_set_drops_the_fingerprint():
    """use_fingerprint=False gives a small dense set (no Morgan), which needs no PCA."""
    model = TabularFoundationModel(n_bits=256, use_fingerprint=False, pca_components=500)
    model._fit_reducer(model._raw_features(SMILES))
    width = model._features(SMILES).shape[1]
    assert width == model._raw_features(SMILES).shape[1]  # passthrough, no PCA
    assert width < 256  # the 256 fingerprint bits are absent


class _Recorder:
    """A stand-in regressor that records the labels it was fit on (no TabPFN)."""

    def fit(self, features, y):
        self.features = np.asarray(features)
        self.y = np.asarray(y, dtype=float)
        return self

    def predict(self, features):
        return np.zeros(len(features))


def test_fit_uses_only_uncensored_reference_assay_rows(monkeypatch):
    """F09 guard: screen pseudo-labels, censored rows and other assays are dropped from the fit,
    so the tabular model trains only on uncensored reference-assay measurements."""
    recorder = _Recorder()
    model = TabularFoundationModel(n_bits=64, pca_components=10_000)
    monkeypatch.setattr(model, "_regressor", lambda: recorder)

    def row(smiles, assay, value, censored=False):
        return {
            "SMILES": smiles,
            "target": "CYP1A2",
            "assay": assay,
            "value": value,
            "conf_low": value,
            "conf_high": value,
            "weight": 1.0,
            "is_censored": censored,
        }

    observations = pd.DataFrame(
        [
            row("CCO", "direct_inhibition_drc", 5.0),
            row("c1ccccc1", "direct_inhibition_drc", 6.0),
            row("CCN", "direct_inhibition_screen", 9.0),  # screen pseudo-label: dropped
            row("CCO", "tdi_drc", 8.0),  # preincubated assay: dropped
            row("CC(=O)O", "direct_inhibition_drc", 3.0, censored=True),  # censored: dropped
        ]
    )
    model.fit(observations)
    assert sorted(recorder.y.tolist()) == [5.0, 6.0]


def test_auxiliary_potency_rows_join_the_fit_behind_one_indicator_column_per_assay(monkeypatch):
    recorder = _Recorder()
    model = TabularFoundationModel(n_bits=64, pca_components=10_000)
    monkeypatch.setattr(model, "_regressor", lambda: recorder)

    def row(smiles, assay, value, target="CYP1A2", censored=False):
        return {
            "SMILES": smiles,
            "target": target,
            "assay": assay,
            "value": value,
            "is_censored": censored,
        }

    observations = pd.DataFrame(
        [
            row("CCO", "direct_inhibition_drc", 5.0),
            row("c1ccccc1", "direct_inhibition_drc", 6.0),
            row("CCN", "fluorogenic_pic50", 7.0),
            row("CC(=O)O", "luminescent_ac50", 4.0, censored=True),  # censored: dropped
            row("CCCC", "luminescent_max_response", 0.5),  # a response, not a potency: dropped
            row("CCCCC", "fluorogenic_pic50", 8.0, target="CYP2D6"),  # another endpoint
            row("CCCCCC", "luminescent_ac50", 3.0, target="CYP2C19"),  # not a scored endpoint
        ]
    )
    featurised = []
    raw_features = model._raw_features
    monkeypatch.setattr(
        model, "_raw_features", lambda smiles: featurised.extend(smiles) or raw_features(smiles)
    )
    model.fit(observations)
    assert "CCCCCC" not in featurised
    assert sorted(recorder.y.tolist()) == [5.0, 6.0, 7.0]
    assert model._aux_assays == ("fluorogenic_pic50",)
    width = model._raw_features(["CCO"]).shape[1]
    assert recorder.features.shape == (3, width + 1)
    indicator = dict(zip(recorder.y, recorder.features[:, -1], strict=True))
    assert indicator == {5.0: 0.0, 6.0: 0.0, 7.0: 1.0}
    assert model._features(["CCO"])[0, -1] == 0.0  # a scored compound is on no auxiliary assay
