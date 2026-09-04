"""Route A: the ligand-only baseline.

One model per endpoint, fitted on the compounds where that endpoint was measured.
Fitting per endpoint rather than as one multi-output model is deliberate: coverage
runs 26-48% and is not missing at random, and the isoforms do not share a mechanism
uniformly -- CYP2C9 and CYP3A4 correlate at Spearman 0.69 while CYP2D6 correlates
with everything at 0.06 or below.

The ensemble exists for its *spread*, not its mean. Seed averaging is known to add
almost nothing to accuracy here, but the decision layer needs a distribution to
integrate the dead-zone loss against, and disagreement between independently seeded
fits is the cheapest honest source of one.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)

MIN_LABELS_PER_ENDPOINT = 50


class LigandOnlyRoute:
    """Fingerprint and descriptor ensemble over SMILES alone."""

    name = "ligand"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        n_members: int = 5,
        n_estimators: int = 300,
        min_samples_leaf: int = 2,
        n_bits: int = 2048,
        seed: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.endpoints = endpoints
        self.n_members = n_members
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_bits = n_bits
        self.seed = seed
        self.n_jobs = n_jobs
        self._members: dict[str, list[RandomForestRegressor]] = {}

    def _features(self, smiles: list[str]) -> NDArray[np.float32]:
        return featurise(smiles, n_bits=self.n_bits)

    def fit(self, smiles: list[str], targets: NDArray[np.float64]) -> "LigandOnlyRoute":
        """Fit one ensemble per endpoint on that endpoint's measured compounds.

        Args:
            smiles: Training compounds.
            targets: Shape ``(n_compounds, n_endpoints)``, NaN where unmeasured.

        Raises:
            ValueError: If an endpoint has fewer than ``MIN_LABELS_PER_ENDPOINT``
                measured compounds, which would make its ensemble meaningless.
        """
        if targets.shape != (len(smiles), len(self.endpoints)):
            raise ValueError(f"targets must be ({len(smiles)}, {len(self.endpoints)})")
        features = self._features(smiles)

        self._members = {}
        for column, endpoint in enumerate(self.endpoints):
            measured = np.isfinite(targets[:, column])
            if measured.sum() < MIN_LABELS_PER_ENDPOINT:
                raise ValueError(
                    f"{endpoint}: only {measured.sum()} labels, need {MIN_LABELS_PER_ENDPOINT}"
                )
            logger.info("fitting %s on %d compounds", endpoint, measured.sum())
            self._members[endpoint] = [
                RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    n_jobs=self.n_jobs,
                    random_state=self.seed + member,
                ).fit(features[measured], targets[measured, column])
                for member in range(self.n_members)
            ]
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict a distribution for every compound and endpoint."""
        if not self._members:
            raise RuntimeError("route is not fitted")
        features = self._features(smiles)
        samples = np.stack(
            [
                np.column_stack(
                    [
                        self._members[endpoint][member].predict(features)
                        for endpoint in self.endpoints
                    ]
                )
                for member in range(self.n_members)
            ]
        )
        return PredictiveDistribution(samples, self.endpoints)
