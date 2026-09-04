"""Choosing which model family to trust, per compound. **Measured, and it does not work.**

Kept as a recorded negative result rather than deleted, because the idea is an obvious
one to have twice.

The motivation was an oracle: picking the best family for each compound scored 0.4184
macro ST-RAE against 0.6891 for the best single family, apparently 39% of headroom
going spare. Two ways of claiming it were built and both lost to simply using the best
family:

============================  ==========
best single family (boosted)  **0.6891**
router, probability blend     0.7163
router, hard argmax           0.7565
oracle, using labels          0.4184
============================  ==========

The oracle number is an artefact, which is why. Taking the minimum absolute error across
k models is biased low by construction: three simulated models with *identical* accuracy
and independent errors show a 56-58% apparent headroom, larger than the 39% seen here.
The real gap is smaller than pure noise would produce, which says the families' errors
are correlated -- they share featurisation and training data -- and that there is less
than nothing to route.

Cross-model disagreement fails for the same reason: it correlates with the best family's
absolute error at Spearman +0.088. Averaging the families also loses, at 0.7049, since
it drags a good model toward two worse ones.

So: an oracle-over-k gap is not evidence of exploitable structure, and should be checked
against a noise simulation before anyone builds on it. The machinery below is correct and
tested; the premise it serves is not supported.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.features.molecular import featurise

logger = logging.getLogger(__name__)

ROUTING_STRATEGIES = ("argmax", "blend")


@dataclass
class Router:
    """Predicts which model family will be most accurate for a compound."""

    families: tuple[str, ...]
    strategy: str = "blend"
    n_bits: int = 2048
    seed: int = 0
    classifier: HistGradientBoostingClassifier | None = field(default=None, repr=False)

    def fit(self, smiles: list[str], winner: NDArray[np.int64]) -> "Router":
        """Learn the winning family from structure.

        Args:
            smiles: Training compounds.
            winner: Index into :attr:`families` of the family with the smallest error.
        """
        if len(set(winner.tolist())) < 2:
            raise ValueError("every compound has the same winning family; nothing to route")
        x = featurise(smiles, n_bits=self.n_bits)
        self.classifier = HistGradientBoostingClassifier(random_state=self.seed, max_iter=200)
        self.classifier.fit(x, winner)
        counts = np.bincount(winner, minlength=len(self.families))
        logger.info(
            "router trained on %d compounds; wins per family: %s",
            len(smiles),
            dict(zip(self.families, counts.tolist(), strict=True)),
        )
        return self

    def weights(self, smiles: list[str]) -> NDArray[np.float64]:
        """Per-compound weight over families, shape ``(n_compounds, n_families)``."""
        if self.classifier is None:
            raise RuntimeError("router is not fitted")
        probability = self.classifier.predict_proba(featurise(smiles, n_bits=self.n_bits))
        # predict_proba only spans the classes actually seen in training
        full = np.zeros((len(smiles), len(self.families)))
        full[:, self.classifier.classes_] = probability
        if self.strategy == "argmax":
            hard = np.zeros_like(full)
            hard[np.arange(len(full)), full.argmax(axis=1)] = 1.0
            return hard
        return full

    def combine(
        self, smiles: list[str], predictions: dict[str, NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """Route or blend the families' predictions for these compounds."""
        weight = self.weights(smiles)
        stacked = np.stack([predictions[name] for name in self.families])
        return np.einsum("fce,cf->ce", stacked, weight)


def out_of_fold_predictions(
    fit_predict: dict[str, object],
    frame: pd.DataFrame,
    build_observations: object,
    n_splits: int = 3,
    seed: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Predictions for every row, made by a model that did not train on that row.

    Scoring a family on compounds it trained on would make the router prefer whichever
    family memorises hardest, which is the opposite of what it should learn.
    """
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    collected = {
        name: np.full((len(frame), len(REGRESSION_ENDPOINTS)), np.nan) for name in fit_predict
    }
    for index, (train_rows, test_rows) in enumerate(folds.split(frame)):
        observations = build_observations(frame.iloc[train_rows])
        smiles = frame.iloc[test_rows]["SMILES"].tolist()
        for name, build in fit_predict.items():
            model = build()
            model.fit(observations)
            collected[name][test_rows] = model.predict(smiles).mean
        logger.info("fold %d of %d complete", index + 1, n_splits)
    return collected


def winning_family(
    predictions: dict[str, NDArray[np.float64]], truth: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Which family was closest, per compound, averaged over its labelled endpoints.

    Returns the winner index and a mask of compounds with at least one label. A
    compound with no label cannot say which family won and is excluded rather than
    given an arbitrary winner.
    """
    names = list(predictions)
    stacked = np.stack([np.abs(predictions[n] - truth) for n in names])
    labelled = ~np.isnan(truth)
    usable = labelled.any(axis=1)
    with np.errstate(invalid="ignore"):
        mean_error = np.where(labelled[None], stacked, np.nan)
        mean_error = np.nanmean(mean_error, axis=2)
    winner = np.nanargmin(np.where(np.isnan(mean_error), np.inf, mean_error), axis=0)
    return winner[usable].astype(np.int64), usable
