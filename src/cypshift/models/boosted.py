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
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import (
    ASSAY_FLOOR_PIC50,
    ISOFORMS,
    REGRESSION_ENDPOINTS,
    TDI_SHIFT_THRESHOLD,
)
from cypshift.data.assembly import ASSAYS, encode_conditions
from cypshift.features.embeddings import block_for
from cypshift.features.molecular import featurise
from cypshift.models.base import (
    PredictiveDistribution,
    active_stages,
    rows_for_stage,
    training_stages,
)
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
        physchem: bool = True,
        mechanism: bool = True,
        fingerprint_radius: int = 2,
        use_structure: bool = False,
        structure_cache: str = "data/structure",
        surrogate: object | None = None,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        n_members: int = 5,
        loss: str = "soft_threshold",
        seed: int = 0,
        training: object | None = None,
        molecular_embedding: str = "none",
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.conditioning = conditioning
        self.n_bits = n_bits
        #: Which molecular feature blocks to include and the fingerprint radius.
        #: These are read by featurise; toggling physchem changes the dense-block width,
        #: which the scaler follows. Swept via features.* in the config.
        self.physchem = physchem
        self.mechanism = mechanism
        self.fingerprint_radius = fingerprint_radius
        #: Append cached Boltz-2 cofold features per (compound, enzyme). See
        #: :mod:`cypshift.structure`.
        self.use_structure = use_structure
        self.structure_cache = structure_cache
        #: Optional callable appended to the feature matrix, used to feed the
        #: single-concentration screen in as an input rather than as extra training
        #: rows. See :mod:`cypshift.data.surrogate`.
        self.surrogate = surrogate
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.n_members = n_members
        self.loss = loss
        self.seed = seed
        #: Training curriculum (an object exposing ``.stages``), or None for one pooled
        #: stage. The tree has no trunk, so a multi-stage curriculum continues boosting
        #: from the previous stage's members rather than freezing anything.
        self.training = training
        #: Which cached foundation-model embedding to append, or "none".
        self.molecular_embedding = molecular_embedding
        self._members: list[lgb.LGBMRegressor] = []
        self._offset = 0.0

    def _features(self, smiles: list[str], encoding: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.hstack(
            [
                featurise(
                    smiles,
                    n_bits=self.n_bits,
                    radius=self.fingerprint_radius,
                    physchem=self.physchem,
                    mechanism=self.mechanism,
                    extra=self.surrogate,
                    embedding=block_for(self.molecular_embedding),
                ),
                encoding,
            ]
        ).astype(np.float32)

    def _conditioning(self, frame: pd.DataFrame) -> NDArray[np.float32]:
        """The per-row conditioning block: the one-hot, plus cofold features when on."""
        encoding = encode_conditions(frame, self.conditioning, self.targets, self.assays)
        if not self.use_structure:
            return encoding
        from cypshift.structure.features import structure_block

        block = structure_block(frame, Path(self.structure_cache))
        return np.hstack([encoding, block]).astype(np.float32)

    def fit(self, observations: pd.DataFrame) -> "BoostedTrees":
        """Train on the observation table, using each row's own credible interval.

        Rows without bounds fall back to their point value, which makes the objective
        plain absolute error for them.
        """
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        encoding = self._conditioning(usable)
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

        # A curriculum trains the stages in order; the tree has no trunk to freeze, so a
        # later stage continues boosting from the previous stage's trees (init_model). One
        # pooled stage over every tier reduces to a single fit, the previous behaviour.
        stages = active_stages(
            usable, training_stages(self.training, self.n_estimators, self.learning_rate)
        )
        self._members = []
        for member in range(self.n_members):
            booster = None
            for stage in stages:
                mask = rows_for_stage(usable, stage)
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
                        lower[mask] - self._offset, upper[mask] - self._offset
                    )
                elif self.loss in ("squared", "absolute"):
                    parameters["objective"] = (
                        "regression" if self.loss == "squared" else "regression_l1"
                    )
                else:
                    raise ValueError(f"unknown objective {self.loss!r}")
                model = lgb.LGBMRegressor(**parameters)
                model.fit(
                    features[mask],
                    value[mask] - self._offset,
                    sample_weight=weight[mask],
                    init_model=booster,
                )
                booster = model.booster_
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
            query["SMILES"] = smiles
            features = self._features(smiles, self._conditioning(query))
            for index, member in enumerate(self._members):
                samples[index, :, column] = member.predict(features) + self._offset
        return PredictiveDistribution(samples, self.endpoints)

    def _predict_pic50_at(self, smiles: list[str], enzyme: str, assay: str) -> NDArray[np.float64]:
        """Ensemble-mean pIC50 for one enzyme under one assay condition."""
        query = pd.DataFrame(
            {"target": [enzyme] * len(smiles), "assay": [assay] * len(smiles), "SMILES": smiles}
        )
        features = self._features(smiles, self._conditioning(query))
        return np.mean([m.predict(features) + self._offset for m in self._members], axis=0)

    def predict_is_time_dependent(
        self, smiles: list[str], assay_floor: float = ASSAY_FLOOR_PIC50
    ) -> NDArray[np.bool_]:
        """Time-dependent-inhibition call from *two scalar values* rather than two curves.

        The scalar counterpart of the curve model's comparison: predict the direct and
        preincubated pIC50 for each enzyme and apply the organisers' rule. Requires the
        preincubated arm in training (``sources.tdi_arm``), or the tdi-assay prediction
        is untrained. Returns one column per isoform, matching the curve model's shape.
        """
        flags = []
        for enzyme in (e.split("_")[0] for e in REGRESSION_ENDPOINTS):
            direct = self._predict_pic50_at(smiles, enzyme, "direct_inhibition_drc")
            preincubated = self._predict_pic50_at(smiles, enzyme, "tdi_drc")
            flags.append(
                np.where(
                    direct > assay_floor,
                    (preincubated - direct) > TDI_SHIFT_THRESHOLD,
                    preincubated > assay_floor + TDI_SHIFT_THRESHOLD,
                )
            )
        return np.stack(flags, axis=1)
