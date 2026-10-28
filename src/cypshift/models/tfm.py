"""A tabular foundation model (TabPFN) as a per-endpoint regressor.

TabPFN is an in-context transformer that fits a small table in a single forward pass. This is
only the *model*: it consumes the **same configured feature set as every other model**
(fingerprint + physicochemical + mechanistic + ionization + embedding + screen_model), so swapping
``model.kind`` from boosted to tfm changes the model and nothing else. TabPFN has a feature
ceiling (~2000 columns) and does poorly on raw high-dimensional sparse fingerprints, so its one
adaptation is a dimensionality module -- impute, standardise, then PCA the assembled matrix down
to ``pca_components`` -- fit on the training compounds and applied unchanged to the test set. The
adaptation compresses the shared feature set; it does not substitute a different one.

It fits one regressor per endpoint, on the compounds that carry that label, which sidesteps the
sparse multi-label coverage. Auxiliary potency rows of the same endpoint (Octant, Tox21, ...) join
the context with one indicator column per auxiliary assay, so the regressor can place each assay on
its own scale; the scored compounds carry zeros in those columns. It has no sample weights and no
custom objective, so it cannot see the interval-censored loss; it is meant as a blend member whose
different inductive bias complements the boosted/graph families. The backend is a parameter:
``tabpfn`` (in-process) or ``mitra`` (Amazon's tabular foundation model, Apache-2.0).

TabPFN needs the ``tfm`` extra (``pip install cypshift[tfm]``). Mitra comes with AutoGluon, whose
pandas/pyarrow pins the main environment cannot take, so it runs in its own environment
(``MITRA_PYTHON``, default ``/mnt/wsl/ext/envs/mitra/bin/python``): ``fit`` keeps the context
rows, and ``predict`` runs :mod:`cypshift.models.mitra_worker` there once for every endpoint.
Both download their weights to the HuggingFace cache; ``HF_HOME`` is pointed at the shared ext
cache if unset.
"""

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import REFERENCE_ASSAY
from cypshift.features.assembly import molecular_feature_matrix
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_HOME", "/mnt/wsl/ext/hf_cache")
TFM_BACKENDS = ("tabpfn", "mitra")
DEFAULT_MITRA_PYTHON = "/mnt/wsl/ext/envs/mitra/bin/python"
MITRA_WORKER = Path(__file__).with_name("mitra_worker.py")
#: The auxiliary assays whose rows are potencies on the scored endpoint's own axis. Screen,
#: preincubated and response rows are different quantities and never enter the context.
POTENCY_ASSAYS = ("fluorogenic_pic50", "luminescent_ac50", "literature_ic50", "reversible_pic50")


class _ContextRows:
    """The fit rows of one endpoint, kept for a backend that conditions on them at predict time
    (Mitra runs in another environment, so nothing is trained until ``predict``)."""

    def fit(self, features: NDArray[np.float32], labels: NDArray[np.float64]) -> "_ContextRows":
        self.features = features
        self.labels = labels
        return self


class TabularFoundationModel:
    """One tabular-foundation-model regressor per endpoint over the shared feature set."""

    name = "tabular_foundation_model"
    endpoints = tuple(REGRESSION_ENDPOINTS)

    def __init__(
        self,
        targets: tuple[str, ...] = ISOFORMS,
        backend: str = "tabpfn",
        n_bits: int = 2048,
        fingerprint_radius: int = 2,
        use_fingerprint: bool = True,
        physchem: bool = True,
        mechanism: bool = True,
        molecular_embedding: str = "none",
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
        screen_model: object | None = None,
        pca_components: int = 500,
        fine_tune: bool = True,
        fine_tune_steps: int = 50,
        n_estimators: int = 0,
        softmax_temperature: float = 0.0,
        prediction: str = "mean",
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        if backend not in TFM_BACKENDS:
            raise ValueError(f"unknown tfm backend {backend!r}; have {TFM_BACKENDS}")
        if prediction not in ("mean", "median"):
            raise ValueError(f"unknown tfm prediction {prediction!r}; have mean, median")
        self.targets = targets
        self.backend = backend
        self.n_bits = n_bits
        self.fingerprint_radius = fingerprint_radius
        self.use_fingerprint = use_fingerprint
        self.physchem = physchem
        self.mechanism = mechanism
        self.molecular_embedding = molecular_embedding
        self.ionization_geometry = ionization_geometry
        self.ionization_cache = ionization_cache
        self.screen_model = screen_model
        #: PCA target width for the dimensionality adaptation (TabPFN's feature ceiling is
        #: ~2000 and it degrades on raw sparse fingerprints). Swept.
        self.pca_components = pca_components
        #: Mitra only: fine-tune the pretrained weights on the context rows before predicting, for
        #: this many steps (AutoGluon's own defaults).
        self.fine_tune = fine_tune
        self.fine_tune_steps = fine_tune_steps
        #: TabPFN only; 0 leaves its own default ("auto" estimators, the checkpoint's temperature).
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.prediction = prediction
        self.device = device
        self.seed = seed
        self._models: dict[str, object] = {}
        self._reducer: object | None = None
        #: The auxiliary assays that got an indicator column at fit time, in column order.
        self._aux_assays: tuple[str, ...] = ()

    def _raw_features(self, smiles: Sequence[str]) -> NDArray[np.float32]:
        """The shared configured feature set -- identical assembly to the other models."""
        return molecular_feature_matrix(
            smiles,
            n_bits=self.n_bits,
            fingerprint_radius=self.fingerprint_radius,
            use_fingerprint=self.use_fingerprint,
            physchem=self.physchem,
            mechanism=self.mechanism,
            screen_model=self.screen_model,
            ionization_geometry=self.ionization_geometry,
            ionization_cache=self.ionization_cache,
            molecular_embedding=self.molecular_embedding,
        )

    def _features(self, smiles: Sequence[str]) -> NDArray[np.float32]:
        """The shared feature set compressed by the fitted impute/scale/PCA adaptation, followed
        by zeros for the auxiliary-assay indicators (a scored compound is on no auxiliary assay)."""
        if self._reducer is None:
            raise RuntimeError("feature reducer not fitted; call fit first")
        reduced = self._reducer.transform(self._raw_features(smiles)).astype(np.float32)
        return np.hstack([reduced, np.zeros((len(reduced), len(self._aux_assays)), np.float32)])

    def _device(self) -> str:
        import torch

        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _regressor(self) -> object:
        if self.backend == "mitra":
            return _ContextRows()
        from tabpfn import TabPFNRegressor

        options: dict[str, object] = {}
        if self.n_estimators:
            options["n_estimators"] = self.n_estimators
        if self.softmax_temperature:
            options["softmax_temperature"] = self.softmax_temperature
        return TabPFNRegressor(
            device=self._device(),
            random_state=self.seed,
            ignore_pretraining_limits=True,
            **options,
        )

    def _fit_reducer(self, raw: NDArray[np.float32]) -> None:
        """Fit the impute/scale/PCA adaptation on the training compounds' raw features only.

        PCA keeps at most ``pca_components`` components, capped by the data's own rank so a
        narrow feature set or few compounds never asks for more than exists.
        """
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        # PCA only when the assembled set exceeds the target width (e.g. the fingerprint is on);
        # a dense-only set already under the ceiling is passed through, the input TabPFN prefers.
        if raw.shape[1] > self.pca_components:
            width = min(self.pca_components, raw.shape[0], raw.shape[1])
            steps.append(("pca", PCA(n_components=width, random_state=self.seed)))
            logger.info("tfm reducer: %d features -> %d PCA components", raw.shape[1], width)
        else:
            logger.info("tfm reducer: %d features, no PCA (already within ceiling)", raw.shape[1])
        self._reducer = Pipeline(steps).fit(raw)

    def fit(self, observations: pd.DataFrame) -> "TabularFoundationModel":
        """Fit one regressor per endpoint on that endpoint's uncensored rows of the reference
        assay and of the auxiliary potency assays.

        The backend has no sample weights, censoring or assay conditioning. Screen
        pseudo-labels, the preincubated assay and response rows would enter as interchangeable,
        equally-weighted measurements, so they are left out; each auxiliary potency assay gets
        an indicator column instead. The pipeline rejects the settings that would add the
        excluded rows for this model (see ExperimentConfig.validate).
        """
        usable = observations[np.isfinite(observations["value"])]
        if "is_censored" in usable.columns:
            usable = usable[~usable["is_censored"].astype(bool)]
        if "assay" in usable.columns:
            usable = usable[usable["assay"].isin((REFERENCE_ASSAY, *POTENCY_ASSAYS))]
        usable = usable[usable["target"].isin(self.targets)]  # related enzymes' rows are not read
        if usable[usable["assay"] == REFERENCE_ASSAY].empty:
            raise ValueError("no uncensored reference-assay rows to fit the tabular model on")
        self._aux_assays = tuple(
            assay for assay in POTENCY_ASSAYS if (usable["assay"] == assay).any()
        )
        unique = usable["SMILES"].drop_duplicates().tolist()
        raw = self._raw_features(unique)
        self._fit_reducer(raw)
        reduced = self._reducer.transform(raw).astype(np.float32)
        position = {smiles: i for i, smiles in enumerate(unique)}
        for isoform in self.targets:
            rows = usable[usable["target"] == isoform]
            if rows[rows["assay"] == REFERENCE_ASSAY].empty:
                continue
            indicators = np.column_stack(
                [(rows["assay"] == assay).to_numpy(np.float32) for assay in self._aux_assays]
                or [np.zeros((len(rows), 0), np.float32)]
            )
            features = np.hstack([reduced[[position[s] for s in rows["SMILES"]]], indicators])
            model = self._regressor()
            model.fit(features, rows["value"].to_numpy(float))
            self._models[isoform] = model
            logger.info(
                "tfm %s: fit on %d rows (%d auxiliary), %d features",
                isoform,
                len(rows),
                int((rows["assay"] != REFERENCE_ASSAY).sum()),
                features.shape[1],
            )
        if not self._models:
            raise ValueError("no endpoint had finite labels to fit on")
        return self

    def _predict_mitra(self, features: NDArray[np.float32]) -> dict[str, NDArray[np.float64]]:
        """Every endpoint's Mitra predictions of ``features``, from one run of the worker in the
        Mitra environment."""
        settings = {
            "fine_tune": self.fine_tune,
            "fine_tune_steps": self.fine_tune_steps,
            "device": self._device(),
            "seed": self.seed,
        }
        with tempfile.TemporaryDirectory(prefix="mitra_") as directory:
            source, target = Path(directory, "source.npz"), Path(directory, "target.npz")
            arrays = {"query": features}
            for isoform, context in self._models.items():
                arrays[f"x__{isoform}"] = context.features
                arrays[f"y__{isoform}"] = context.labels
            np.savez(source, **arrays)
            command = [
                os.environ.get("MITRA_PYTHON", DEFAULT_MITRA_PYTHON),
                str(MITRA_WORKER),
                "--source",
                str(source),
                "--target",
                str(target),
                "--settings",
                json.dumps(settings),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"the Mitra worker exited {result.returncode}:\n{result.stderr[-2000:]}"
                )
            logger.info("mitra worker:\n%s", result.stderr.strip())
            with np.load(target) as predictions:
                return {isoform: predictions[f"prediction__{isoform}"] for isoform in self._models}

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict every endpoint; carried as a single-member predictive distribution."""
        features = self._features(smiles)
        if self.backend == "mitra":
            predicted = self._predict_mitra(features)
        else:
            options = {} if self.prediction == "mean" else {"output_type": self.prediction}
            predicted = {
                isoform: model.predict(features, **options)
                for isoform, model in self._models.items()
            }
        columns = []
        for endpoint in REGRESSION_ENDPOINTS:
            isoform = endpoint.split("_")[0]
            columns.append(predicted.get(isoform, np.full(len(smiles), np.nan)))
        samples = np.column_stack(columns)[None, :, :]
        return PredictiveDistribution(samples.astype(np.float64), self.endpoints)
