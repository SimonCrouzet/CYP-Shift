"""Model A: one model over all isoforms, with target and assay as input columns.

A row is a single measurement, and its input is the molecule's features plus a
one-hot vector naming which protein it was measured against and under which
experiment. That is all "context" means here -- extra input columns, not a
separate pathway. Aspirin measured on CYP3A4 and on CYP1A2 gives identical molecular
features, so those columns are the only thing distinguishing the two rows, which is
what lets one model serve every isoform instead of four models sharing nothing.

That is what "structure-free" buys over "ligand-only". No 3D structure is used, but
the protein and the experiment are both in the input, so a measurement made on a
different enzyme or under a different incubation still trains the shared
representation. Four independent per-isoform models transfer nothing between them and
cannot use a single-concentration screen reading at all.

Prediction is always made at the scored condition, ``direct_inhibition_drc``, whatever
mixture of assays the model was trained on.
"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import ASSAYS, context_columns
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.ensemble import DEFAULT_MEMBERS, MemberSpec

logger = logging.getLogger(__name__)

SCORED_ASSAY = "direct_inhibition_drc"


class SharedEnsemble:
    """Structure-free model over a long observation table."""

    name = "structure_free"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        members: tuple[MemberSpec, ...] = DEFAULT_MEMBERS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        n_bits: int = 2048,
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.members = members
        self.targets = targets
        self.assays = assays
        self.n_bits = n_bits
        self.seed = seed
        self._fitted: list = []

    def _molecular(self, smiles: list[str], spec: MemberSpec) -> NDArray[np.float32]:
        matrix = featurise(
            smiles,
            fingerprint=spec.fingerprint,
            physchem=spec.physchem,
            mechanism=spec.mechanism,
            n_bits=self.n_bits,
        )
        return (matrix > 0).astype(np.float32) if spec.name.startswith("nearest") else matrix

    def fit(self, observations: pd.DataFrame) -> "SharedEnsemble":
        """Fit every member on the whole observation table at once.

        Args:
            observations: Long table from
                :func:`cypshift.data.assembly.build_observations`.

        Raises:
            ValueError: If the table carries no usable measurement.
        """
        usable = observations[np.isfinite(observations["value"])]
        if usable.empty:
            raise ValueError("no finite measurements to fit on")
        context = context_columns(usable, self.targets, self.assays)
        smiles = usable["SMILES"].tolist()
        weights = usable["weight"].to_numpy(float)
        values = usable["value"].to_numpy(float)
        logger.info(
            "fitting %d members on %d observations across %d targets and %d assays",
            len(self.members),
            len(usable),
            usable["target"].nunique(),
            usable["assay"].nunique(),
        )

        self._fitted = []
        for spec in self.members:
            features = np.hstack([self._molecular(smiles, spec), context])
            estimator = spec.build(self.seed)
            try:
                estimator.fit(features, values, sample_weight=weights)
            except TypeError:
                # Neighbour models take no sample weights; the source mixture is then
                # carried by the context block alone.
                estimator.fit(features, values)
            self._fitted.append(estimator)
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict at the scored assay condition for every endpoint."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        n = len(smiles)
        samples = np.empty((len(self.members), n, len(self.endpoints)))
        for column, endpoint in enumerate(self.endpoints):
            target = endpoint.split("_")[0]
            query = pd.DataFrame({"target": [target] * n, "assay": [SCORED_ASSAY] * n})
            context = context_columns(query, self.targets, self.assays)
            for index, spec in enumerate(self.members):
                features = np.hstack([self._molecular(smiles, spec), context])
                samples[index, :, column] = self._fitted[index].predict(features)
        return PredictiveDistribution(samples, self.endpoints)
