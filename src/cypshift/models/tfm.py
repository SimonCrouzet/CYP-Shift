"""A tabular foundation model (TabPFN) as a per-endpoint regressor.

TabPFN is an in-context transformer that fits a small table in a single forward pass. The
molecular evidence is clear that it does well on **dense** inputs (learned embeddings, RDKit
descriptors) and poorly on raw high-dimensional sparse fingerprints, so this model deliberately
drops the Morgan block and feeds only the physicochemical, mechanistic, ionization and optional
foundation-model-embedding blocks. It fits one regressor per endpoint, on the compounds that
carry that label, which also sidesteps the sparse multi-label coverage -- no imputation.

It has no sample weights and no custom objective, so it cannot see the interval-censored loss;
it is meant as a blend member whose different inductive bias complements the boosted/graph
families, not as the censoring-aware model. Backend is a parameter so a second tabular
foundation model (e.g. Mitra) can slot in behind the same interface later.

Needs the ``tfm`` extra (``pip install cypshift[tfm]``) and downloads its weights to the
HuggingFace cache; ``HF_HOME`` is pointed at the shared ext cache if unset.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.features.embeddings import block_for
from cypshift.features.ionization import block_for as ionization_block_for
from cypshift.features.molecular import featurise
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
        physchem: bool = True,
        mechanism: bool = True,
        molecular_embedding: str = "none",
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        if backend not in TFM_BACKENDS:
            raise ValueError(f"unknown tfm backend {backend!r}; have {TFM_BACKENDS}")
        self.targets = targets
        self.backend = backend
        self.physchem = physchem
        self.mechanism = mechanism
        self.molecular_embedding = molecular_embedding
        self.ionization_geometry = ionization_geometry
        self.ionization_cache = ionization_cache
        self.device = device
        self.seed = seed
        self._models: dict[str, object] = {}

    def _features(self, smiles: Sequence[str]) -> NDArray[np.float32]:
        """Dense feature blocks only -- no Morgan, which TabPFN handles poorly."""
        return featurise(
            list(smiles),
            fingerprint=False,
            physchem=self.physchem,
            mechanism=self.mechanism,
            ionization=ionization_block_for(self.ionization_geometry, Path(self.ionization_cache)),
            embedding=block_for(self.molecular_embedding),
        )

    def _regressor(self) -> object:
        import torch
        from tabpfn import TabPFNRegressor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return TabPFNRegressor(device=device, random_state=self.seed)

    def fit(self, observations: pd.DataFrame) -> "TabularFoundationModel":
        """Fit one regressor per endpoint on that endpoint's labelled rows."""
        usable = observations[np.isfinite(observations["value"])]
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
