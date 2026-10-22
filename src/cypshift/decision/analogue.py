"""A local, analogue-based correction layered on top of a global model.

The graph learns one structure-activity function for the whole library. The blinded test,
though, was not drawn from that library: it is ten purchased analogues around each measured
reference hit, so a test compound usually sits near a *known*, same-endpoint training
measurement. A global function cannot use that -- it has already averaged the reference into
its weights -- but a local correction can read it directly:

    y_corrected(x) = (1 - lambda(x)) * y_global(x) + lambda(x) * y_local(x),

where ``y_local`` comes from the query's nearest same-endpoint training neighbours and
``lambda(x)`` is a shrinkage that goes to zero when those neighbours are distant, disagree,
or carry poorly determined activities. Where the query has no real analogue the correction
vanishes and the global prediction stands; where it sits on top of a well-measured reference
the correction speaks loudest. This is the analysis note's avenue 3
(notes/performance_analysis_and_next_steps_2026-10-22.md), and shrinkage is not optional:
the reference hits were selected at the active extreme, so copying their potency inherits
selection noise and misses the very activity cliffs the analogues were bought to probe.

Three local estimators, because which transfers is an empirical question:

``neighbour``
    The similarity-weighted mean of the neighbours' measured values. The plainest baseline.
``kernel``
    A Tanimoto-kernel (Nadaraya-Watson) weighting of the same values, so the closest
    neighbour dominates rather than counting equally with a distant one.
``residual``
    The global prediction plus the similarity-weighted mean of the neighbours' *residuals*
    (measured minus the model's own out-of-fold prediction there). This corrects a local bias
    in the global function rather than discarding it, which is usually the safer move.

Every estimator is interval-aware where it reads a value: given the neighbours' credible
intervals it picks the point minimising the weighted soft-threshold distance to them (the
metric's own loss), not the blind mean of their midpoints. That minimiser of a weighted sum
of soft-threshold absolute errors is piecewise linear in the candidate value, so its optimum
sits at one of the interval bounds and is found by evaluating them.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rdkit import DataStructs

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.data.splits import morgan_fingerprints

logger = logging.getLogger(__name__)

ESTIMATORS = ("neighbour", "kernel", "residual")


def _interval_point(
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """The value minimising the weighted soft-threshold distance to a set of intervals.

    The objective ``sum_i w_i * (max(0, y - high_i) + max(0, low_i - y))`` is convex and
    piecewise linear in ``y``, so a minimiser lies at one of the bounds. Evaluating the
    objective at every bound and taking the best is exact and cheap for the small neighbour
    sets this is called on.
    """
    candidates = np.concatenate([low, high])

    def objective(y: float) -> float:
        above = np.clip(y - high, a_min=0.0, a_max=None)
        below = np.clip(low - y, a_min=0.0, a_max=None)
        return float((weights * (above + below)).sum())

    return float(min(candidates, key=objective))


@dataclass
class AnalogueCorrector:
    """A per-endpoint local correction fit on measured references and their model residuals.

    Fit on the training compounds -- their measured values, credible bounds, and the model's
    own out-of-fold predictions there -- then applied to any query given the global model's
    predictions for it. Nothing here fits on the query's own label, so it transfers to unseen
    analogues rather than memorising a holdout.
    """

    estimator: str = "residual"
    #: Neighbours at or above this Tanimoto are eligible; below it a compound has no analogue.
    min_similarity: float = 0.4
    #: At most this many neighbours per query, the closest first.
    max_neighbours: int = 5
    #: Kernel sharpness for the ``kernel`` estimator and the similarity weights elsewhere:
    #: weight is ``similarity ** power``, so a higher power leans harder on the closest match.
    power: float = 2.0
    #: Pseudo-count in ``lambda = W / (W + shrinkage)``; larger trusts the global model more
    #: until the neighbour evidence (summed weight ``W``) outweighs it.
    shrinkage: float = 1.0
    #: Disagreement (weighted sd of the neighbours' local target, in pIC50) at which the
    #: extra confidence factor ``exp(-disagreement / scale)`` falls to ``1/e``.
    disagreement_scale: float = 1.0
    endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        self._ref_fps: list = []
        # Per endpoint, the reference rows that carry a measurement: fingerprint indices,
        # value, bounds, and the global model's OOF prediction there.
        self._ref: dict[str, dict[str, NDArray[np.float64]]] = {}
        self._ref_index: dict[str, NDArray[np.int_]] = {}

    def fit(
        self,
        reference_smiles: list[str],
        truth: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        model_pred: NDArray[np.float64],
    ) -> "AnalogueCorrector":
        """Record the measured references each query will be corrected against.

        Args:
            reference_smiles: Training compounds, one per row.
            truth: Their measured pIC50, shape ``(n_ref, n_endpoints)``, NaN where unmeasured.
            lower, upper: The credible-interval bounds, same shape; used for the
                interval-aware local point.
            model_pred: The global model's out-of-fold prediction at those compounds, same
                shape, for the ``residual`` estimator.
        """
        self._ref_fps = morgan_fingerprints(reference_smiles)
        self._ref = {}
        self._ref_index = {}
        for column, endpoint in enumerate(self.endpoints):
            measured = np.flatnonzero(np.isfinite(truth[:, column]))
            self._ref_index[endpoint] = measured
            self._ref[endpoint] = {
                "value": truth[measured, column],
                "lower": lower[measured, column],
                "upper": upper[measured, column],
                "residual": truth[measured, column] - model_pred[measured, column],
            }
            logger.info("%s: %d measured references", endpoint, len(measured))
        return self

    def correct(
        self,
        query_smiles: list[str],
        prediction: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the locally corrected prediction, shape ``(n_query, n_endpoints)``.

        ``prediction`` is the global model's output for the same compounds and endpoints. A
        query with no eligible neighbour on an endpoint is returned unchanged there.
        """
        query_fps = morgan_fingerprints(query_smiles)
        corrected = np.array(prediction, dtype=float, copy=True)
        for column, endpoint in enumerate(self.endpoints):
            reference = self._ref[endpoint]
            ref_positions = self._ref_index[endpoint]
            if len(ref_positions) == 0:
                continue
            ref_fps = [self._ref_fps[i] for i in ref_positions]
            for row, fingerprint in enumerate(query_fps):
                similarities = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint, ref_fps))
                corrected[row, column] = self._correct_one(
                    float(prediction[row, column]), similarities, reference
                )
        return corrected

    def _correct_one(
        self,
        global_value: float,
        similarities: NDArray[np.float64],
        reference: dict[str, NDArray[np.float64]],
    ) -> float:
        eligible = np.flatnonzero(similarities >= self.min_similarity)
        if eligible.size == 0:
            return global_value
        # Keep the closest neighbours, and weight them by similarity raised to ``power``.
        order = eligible[np.argsort(-similarities[eligible])][: self.max_neighbours]
        weights = similarities[order] ** self.power
        total = float(weights.sum())
        if total <= 0:
            return global_value

        value = reference["value"][order]
        lower = reference["lower"][order]
        upper = reference["upper"][order]
        residual = reference["residual"][order]

        if self.estimator == "residual":
            local = global_value + float((weights * residual).sum() / total)
            spread_basis = residual
        elif self.estimator == "kernel":
            local = _interval_point(lower, upper, weights)
            spread_basis = value
        else:  # "neighbour": interval-aware weighted point over the neighbours' values
            local = _interval_point(lower, upper, weights)
            spread_basis = value

        # Shrinkage: trust the local estimate in proportion to how much neighbour weight
        # there is and how much the neighbours agree. Both guard against the selection noise
        # the reference hits carry.
        mean = float((weights * spread_basis).sum() / total)
        variance = float((weights * (spread_basis - mean) ** 2).sum() / total)
        disagreement = float(np.sqrt(max(variance, 0.0)))
        lam = (total / (total + self.shrinkage)) * float(
            np.exp(-disagreement / self.disagreement_scale)
        )
        return (1.0 - lam) * global_value + lam * local
