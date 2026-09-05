"""An ensemble whose members disagree for reasons, not because of their seed.

Measured on this dataset, five random forests differing only in random seed agree
with one another to within 0.03 log units while being wrong by 0.45, and their spread
correlates with realised error at 0.03 to 0.07. Every member saw the same data,
through the same features, with the same inductive bias, so they share their failure
modes and their disagreement carries almost no information about where the prediction
is unreliable.

That matters because conformal calibration can correct the *size* of a predictive
interval but cannot create *ordering* that was never in it, and the point-estimate step
needs ordering to do better than shifting every compound by the same amount.

Members here differ along the two axes that actually change what a model can see:
the feature view, since bulk physicochemistry and substructure fingerprints carry
different signal on different isoforms, and the inductive bias, since a neighbour
model and a tree ensemble extrapolate differently outside the training chemistry.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.forest import MIN_LABELS_PER_ENDPOINT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemberSpec:
    """One ensemble member: an estimator and the feature view it is given.

    Attributes:
        name: Identifier used in logs.
        build: Takes a seed, returns an unfitted scikit-learn regressor.
        fingerprint: Give this member the Morgan count fingerprint.
        physchem: Give this member the bulk physicochemical descriptors.
        mechanism: Give this member the mechanistic substructure counts.
    """

    name: str
    build: Callable[[int], object]
    fingerprint: bool = True
    physchem: bool = True
    mechanism: bool = True


#: Five members spanning two feature views and four inductive biases. The neighbour
#: model earns its place despite being the weakest alone: the blinded test set is
#: analog series expanded from training hits, so local transfer is exactly the regime
#: it is good at and the tree ensembles are not.
DEFAULT_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec(
        "random_forest_all",
        lambda seed: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
    ),
    MemberSpec(
        "extra_trees_all",
        lambda seed: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
    ),
    MemberSpec(
        "gradient_boosting_descriptors",
        lambda seed: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=seed
        ),
        fingerprint=False,
    ),
    MemberSpec(
        "random_forest_descriptors",
        lambda seed: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        fingerprint=False,
    ),
    MemberSpec(
        "nearest_neighbours_fingerprint",
        lambda seed: KNeighborsRegressor(n_neighbors=5, weights="distance", metric="jaccard"),
        physchem=False,
        mechanism=False,
    ),
)


class PerTargetEnsemble:
    """Ligand-only model whose members differ in feature view and inductive bias."""

    name = "ligand_diverse"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        members: tuple[MemberSpec, ...] = DEFAULT_MEMBERS,
        n_bits: int = 2048,
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.members = members
        self.n_bits = n_bits
        self.seed = seed
        self._fitted: dict[str, list] = {}

    def _view(self, smiles: list[str], spec: MemberSpec) -> NDArray[np.float32] | NDArray[np.bool_]:
        matrix = featurise(
            smiles,
            fingerprint=spec.fingerprint,
            physchem=spec.physchem,
            mechanism=spec.mechanism,
            n_bits=self.n_bits,
        )
        # Jaccard is a set metric, so the neighbour model gets a genuinely boolean
        # view rather than a float array scikit-learn would silently coerce.
        return (matrix > 0) if spec.name.startswith("nearest") else matrix

    def fit(self, smiles: list[str], targets: NDArray[np.float64]) -> "PerTargetEnsemble":
        """Fit every member on every endpoint's measured compounds."""
        if targets.shape != (len(smiles), len(self.endpoints)):
            raise ValueError(f"targets must be ({len(smiles)}, {len(self.endpoints)})")
        views = {spec.name: self._view(smiles, spec) for spec in self.members}

        self._fitted = {}
        for column, endpoint in enumerate(self.endpoints):
            measured = np.isfinite(targets[:, column])
            if measured.sum() < MIN_LABELS_PER_ENDPOINT:
                raise ValueError(
                    f"{endpoint}: only {measured.sum()} labels, need {MIN_LABELS_PER_ENDPOINT}"
                )
            logger.info(
                "fitting %d members for %s on %d compounds",
                len(self.members),
                endpoint,
                measured.sum(),
            )
            self._fitted[endpoint] = [
                spec.build(self.seed).fit(views[spec.name][measured], targets[measured, column])
                for spec in self.members
            ]
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict a distribution whose members disagree where the models genuinely differ."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        views = {spec.name: self._view(smiles, spec) for spec in self.members}
        samples = np.stack(
            [
                np.column_stack(
                    [
                        self._fitted[endpoint][index].predict(views[spec.name])
                        for endpoint in self.endpoints
                    ]
                )
                for index, spec in enumerate(self.members)
            ]
        )
        return PredictiveDistribution(samples, self.endpoints)
