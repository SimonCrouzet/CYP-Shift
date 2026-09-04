"""Gradient-boosted trees trained on the challenge's own objective.

Two results motivate this combination. Trees beat a feed-forward network on the two
isoforms whose signal is substructural, because a tree splits on a single fingerprint
bit while a dense layer has to learn that as a weighted combination over two thousand
mostly-zero inputs. And the soft-threshold objective beat squared error by 0.115
macro at p = 0.000, far more than any post-hoc adjustment was worth. Until now the
better objective was attached to the weaker model.

The objective is supplied to LightGBM as a custom gradient. Soft-threshold absolute
error is piecewise linear, so its gradient is ±1 outside the credible interval and
exactly zero inside, and its second derivative is zero everywhere. A zero Hessian
gives LightGBM no curvature to work with, so a constant is substituted -- the same
treatment LightGBM's own absolute-error objective uses, where the step size is then
governed by the learning rate rather than by curvature.
"""

import logging
from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import ASSAYS, encode_conditions
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.shared import SCORED_ASSAY

logger = logging.getLogger(__name__)


def soft_threshold_gradient(
    lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> Callable[[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray, NDArray]]:
    """Build LightGBM's gradient and Hessian for soft-threshold absolute error.

    The bounds are captured by closure because LightGBM's objective signature passes
    only the label and the current prediction, and they arrive in training-row order.

    Args:
        lower: Lower credible-interval bound per training row.
        upper: Upper bound per training row.

    Returns:
        A callable LightGBM can use as ``objective``.
    """

    def objective(
        y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        gradient = np.zeros_like(y_pred)
        gradient[y_pred > upper] = 1.0
        gradient[y_pred < lower] = -1.0
        # Constant Hessian: the loss is piecewise linear, so genuine curvature is zero
        # and LightGBM would take unbounded steps.
        return gradient, np.ones_like(y_pred)

    return objective


class BoostedTrees:
    """One boosted model over every isoform, conditioned on target and assay."""

    name = "boosted_trees"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        conditioning: str = "factorised",
        n_bits: int = 2048,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        n_members: int = 5,
        loss: str = "soft_threshold",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.conditioning = conditioning
        self.n_bits = n_bits
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_members = n_members
        self.loss = loss
        self.seed = seed
        self._members: list[lgb.LGBMRegressor] = []
        self._offset = 0.0

    def _features(self, smiles: list[str], encoding: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.hstack([featurise(smiles, n_bits=self.n_bits), encoding]).astype(np.float32)

    def fit(self, observations: pd.DataFrame) -> "BoostedTrees":
        """Train on the observation table, using each row's own credible interval.

        Rows without bounds fall back to their point value, which makes the objective
        plain absolute error for them.
        """
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        encoding = encode_conditions(usable, self.conditioning, self.targets, self.assays)
        features = self._features(usable["SMILES"].tolist(), encoding)
        value = usable["value"].to_numpy(float)
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -1e4, lower)
        weight = usable["weight"].to_numpy(float)

        # A custom objective starts boosting from zero rather than from the label mean,
        # so predictions are offset by it and the trees learn the residual.
        self._offset = float(np.average(value, weights=weight))
        logger.info(
            "fitting %d boosted members on %d observations, objective %s",
            self.n_members,
            len(usable),
            self.loss,
        )

        self._members = []
        for member in range(self.n_members):
            parameters = dict(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.6,
                random_state=self.seed + member,
                verbose=-1,
            )
            if self.loss == "soft_threshold":
                parameters["objective"] = soft_threshold_gradient(
                    lower - self._offset, upper - self._offset
                )
            elif self.loss in ("squared", "absolute"):
                parameters["objective"] = (
                    "regression" if self.loss == "squared" else "regression_l1"
                )
            else:
                raise ValueError(f"unknown objective {self.loss!r}")
            model = lgb.LGBMRegressor(**parameters)
            model.fit(features, value - self._offset, sample_weight=weight)
            self._members.append(model)
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict at the scored assay condition for every endpoint."""
        if not self._members:
            raise RuntimeError("model is not fitted")
        n = len(smiles)
        samples = np.empty((self.n_members, n, len(self.endpoints)))
        for column, endpoint in enumerate(self.endpoints):
            query = pd.DataFrame(
                {"target": [endpoint.split("_")[0]] * n, "assay": [SCORED_ASSAY] * n}
            )
            features = self._features(
                smiles, encode_conditions(query, self.conditioning, self.targets, self.assays)
            )
            for index, member in enumerate(self._members):
                samples[index, :, column] = member.predict(features) + self._offset
        return PredictiveDistribution(samples, self.endpoints)
