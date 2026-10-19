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
sparse multi-label coverage. It has no sample weights and no custom objective, so it cannot see
the interval-censored loss; it is meant as a blend member whose different inductive bias
complements the boosted/graph families. Backend is a parameter so a second tabular foundation
model (e.g. Mitra) can slot in behind the same interface later.

Needs the ``tfm`` extra (``pip install cypshift[tfm]``) and downloads its weights to the
HuggingFace cache; ``HF_HOME`` is pointed at the shared ext cache if unset.
"""

import logging
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.features.assembly import molecular_feature_matrix
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_HOME", "/mnt/wsl/ext/hf_cache")
TFM_BACKENDS = ("tabpfn",)


class TabularFoundationModel:
    """One TabPFN regressor per endpoint over dense (non-fingerprint) features."""

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
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        if backend not in TFM_BACKENDS:
            raise ValueError(f"unknown tfm backend {backend!r}; have {TFM_BACKENDS}")
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
        self.device = device
        self.seed = seed
        self._models: dict[str, object] = {}
        self._reducer: object | None = None

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
        """The shared feature set compressed by the fitted impute/scale/PCA adaptation."""
        if self._reducer is None:
            raise RuntimeError("feature reducer not fitted; call fit first")
        return self._reducer.transform(self._raw_features(smiles)).astype(np.float32)

    def _regressor(self) -> object:
        import torch
        from tabpfn import TabPFNRegressor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return TabPFNRegressor(device=device, random_state=self.seed)

    def _fit_reducer(self, smiles: Sequence[str]) -> None:
        """Fit the impute/scale/PCA adaptation on the training compounds only.

        PCA keeps at most ``pca_components`` components, capped by the data's own rank so a
        narrow feature set or few compounds never asks for more than exists.
        """
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        raw = self._raw_features(smiles)
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
        """Fit one regressor per endpoint on that endpoint's labelled rows."""
        usable = observations[np.isfinite(observations["value"])]
        self._fit_reducer(usable["SMILES"].drop_duplicates().tolist())
        for isoform in self.targets:
            rows = usable[usable["target"] == isoform]
            if rows.empty:
                continue
            features = self._features(rows["SMILES"].tolist())
            model = self._regressor()
            model.fit(features, rows["value"].to_numpy(float))
            self._models[isoform] = model
            logger.info("tfm %s: fit on %d compounds, %d features", isoform, *features.shape)
        if not self._models:
            raise ValueError("no endpoint had finite labels to fit on")
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict every endpoint; carried as a single-member predictive distribution."""
        features = self._features(smiles)
        columns = []
        for endpoint in REGRESSION_ENDPOINTS:
            isoform = endpoint.split("_")[0]
            model = self._models.get(isoform)
            columns.append(
                model.predict(features) if model is not None else np.full(len(smiles), np.nan)
            )
        samples = np.column_stack(columns)[None, :, :]
        return PredictiveDistribution(samples.astype(np.float64), self.endpoints)
