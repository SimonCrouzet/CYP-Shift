"""The tabular foundation model's feature path: the shared set, compressed to fit TabPFN.

These tests cover the modular contract -- that the TFM assembles the same configured feature
set as the other models and only adapts its dimensionality -- without needing TabPFN itself
(the per-endpoint fit, which downloads weights and wants a token, is not exercised here).
"""

import json

import numpy as np
import pandas as pd
import pytest

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


class _MeanRegressor:
    """A stand-in for Mitra that predicts the mean of the labels it was fit on."""

    def __init__(self, settings):
        self.settings = settings

    def fit(self, features, labels):
        self.mean = float(np.mean(labels))
        return self

    def predict(self, features):
        return np.full(len(features), self.mean)


def test_mitra_worker_predicts_each_endpoint_from_its_own_context(tmp_path):
    from cypshift.models import mitra_worker

    source, target = tmp_path / "source.npz", tmp_path / "target.npz"
    np.savez(
        source,
        query=np.zeros((3, 2), np.float32),
        x__CYP1A2=np.zeros((2, 2), np.float32),
        y__CYP1A2=np.array([4.0, 6.0]),
        x__CYP3A4=np.zeros((1, 2), np.float32),
        y__CYP3A4=np.array([8.0]),
    )
    settings = {"fine_tune": False, "fine_tune_steps": 0, "device": "cpu", "seed": 0}
    mitra_worker.run(source, target, settings, make_regressor=_MeanRegressor)
    with np.load(target) as predictions:
        assert sorted(predictions.files) == ["prediction__CYP1A2", "prediction__CYP3A4"]
        assert predictions["prediction__CYP1A2"].tolist() == [5.0, 5.0, 5.0]
        assert predictions["prediction__CYP3A4"].tolist() == [8.0, 8.0, 8.0]


class _PrevalenceClassifier:
    def __init__(self, settings):
        self.settings = settings

    def fit(self, features, labels):
        self.share = float(np.mean(labels))
        assert set(np.unique(labels)) <= {0, 1}
        return self

    def predict_proba(self, features):
        return np.column_stack(
            [np.full(len(features), 1 - self.share)] * 1 + [np.full(len(features), self.share)]
        )


def test_mitra_worker_classification_writes_the_probability_of_the_positive_class(tmp_path):
    from cypshift.models import mitra_worker

    source, target = tmp_path / "source.npz", tmp_path / "target.npz"
    np.savez(
        source,
        query=np.zeros((2, 2), np.float32),
        x__tdi=np.zeros((4, 2), np.float32),
        y__tdi=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    settings = {"task": "classification", "fine_tune": False, "fine_tune_steps": 0, "seed": 0}
    mitra_worker.run(source, target, settings, make_regressor=_PrevalenceClassifier)
    with np.load(target) as predictions:
        assert predictions["prediction__tdi"].tolist() == [0.25, 0.25]


def test_mitra_backend_fits_nothing_in_process_and_predicts_through_the_worker(monkeypatch):
    """fit keeps the context rows; predict hands every endpoint's context and the query to one
    worker run in the Mitra environment and reads the predictions back."""
    import subprocess
    import sys
    from pathlib import Path

    from cypshift.models import mitra_worker, tfm

    model = TabularFoundationModel(
        backend="mitra", n_bits=64, pca_components=10_000, fine_tune=False, fine_tune_steps=7
    )
    observations = pd.DataFrame(
        [
            {
                "SMILES": smiles,
                "target": target,
                "assay": "direct_inhibition_drc",
                "value": value,
                "is_censored": False,
            }
            for target, rows in {
                "CYP1A2": [("CCO", 5.0), ("c1ccccc1", 7.0)],
                "CYP2D6": [("CCN", 3.0)],
            }.items()
            for smiles, value in rows
        ]
    )
    model.fit(observations)
    assert sorted(model._models) == ["CYP1A2", "CYP2D6"]

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        arguments = dict(zip(command[2::2], command[3::2], strict=True))
        mitra_worker.run(
            Path(arguments["--source"]),
            Path(arguments["--target"]),
            json.loads(arguments["--settings"]),
            make_regressor=_MeanRegressor,
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="worker log")

    monkeypatch.setattr(tfm.subprocess, "run", fake_run)
    monkeypatch.setenv("MITRA_PYTHON", sys.executable)
    prediction = model.predict(["CCCC", "CCCCC"])

    assert len(calls) == 1  # one worker run for every endpoint
    assert calls[0][0] == sys.executable
    assert json.loads(calls[0][-1])["fine_tune_steps"] == 7
    by_endpoint = dict(zip(prediction.endpoints, prediction.samples[0].T, strict=True))
    assert by_endpoint["CYP1A2_pIC50_direct_inhibition"].tolist() == [6.0, 6.0]
    assert by_endpoint["CYP2D6_pIC50_direct_inhibition"].tolist() == [3.0, 3.0]
    assert np.isnan(by_endpoint["CYP3A4_pIC50_direct_inhibition"]).all()  # no context


def test_mitra_worker_failure_surfaces_its_error(monkeypatch):
    import subprocess

    from cypshift.models import tfm

    model = TabularFoundationModel(backend="mitra", n_bits=64, pca_components=10_000)
    model._models = {"CYP1A2": tfm._ContextRows().fit(np.zeros((1, 3), np.float32), np.ones(1))}
    model._reducer = type("R", (), {"transform": lambda self, x: x})()
    monkeypatch.setattr(
        model, "_raw_features", lambda smiles: np.zeros((len(smiles), 3), np.float32)
    )
    monkeypatch.setattr(
        tfm.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "CUDA out of memory"),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        model.predict(["CCO"])


def test_fine_tune_settings_belong_to_mitra_only():
    from cypshift.experiment import ExperimentConfig

    config = ExperimentConfig()
    config.override("model.kind", "tfm")
    config.override("model.tfm.fine_tune_steps", 20)
    with pytest.raises(ValueError, match="Mitra's"):
        config.validate()
    config.override("model.tfm.backend", "mitra")
    config.validate()
    config.override("model.tfm.backend", "autogluon")
    with pytest.raises(ValueError, match="model.tfm.backend"):
        config.validate()


def test_tabpfn_settings_reach_the_regressor_only_when_set(monkeypatch):
    import sys
    import types

    seen = []

    class _FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setitem(
        sys.modules, "tabpfn", types.SimpleNamespace(TabPFNRegressor=_FakeTabPFNRegressor)
    )
    TabularFoundationModel(device="cpu")._regressor()
    TabularFoundationModel(device="cpu", n_estimators=32, softmax_temperature=0.7)._regressor()
    assert "n_estimators" not in seen[0] and "softmax_temperature" not in seen[0]
    assert seen[1]["n_estimators"] == 32 and seen[1]["softmax_temperature"] == 0.7


def test_median_prediction_asks_the_regressor_for_the_median_and_mean_leaves_the_call_alone():
    class _Asked:
        def predict(self, features, **options):
            self.options = options
            return np.zeros(len(features))

    model = TabularFoundationModel(n_bits=64, pca_components=10_000)
    model._reducer = type("Identity", (), {"transform": lambda self, raw: raw})()
    model._raw_features = lambda smiles: np.zeros((len(smiles), 3), np.float32)
    model._models = {"CYP1A2": _Asked()}
    model.predict(["CCO"])
    assert model._models["CYP1A2"].options == {}
    model.prediction = "median"
    model.predict(["CCO"])
    assert model._models["CYP1A2"].options == {"output_type": "median"}


def test_tabpfn_settings_belong_to_tabpfn_and_prediction_is_mean_or_median():
    from cypshift.experiment import ExperimentConfig

    config = ExperimentConfig()
    config.override("model.kind", "tfm")
    config.override("model.tfm.n_estimators", 32)
    config.validate()
    config.override("model.tfm.backend", "mitra")
    with pytest.raises(ValueError, match="TabPFN's"):
        config.validate()
    config = ExperimentConfig()
    config.override("model.kind", "tfm")
    config.override("model.tfm.prediction", "mode")
    with pytest.raises(ValueError, match="model.tfm.prediction"):
        config.validate()
