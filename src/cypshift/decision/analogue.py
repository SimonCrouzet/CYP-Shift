"""A local correction added to a global model's prediction from measured training neighbours.

    y_corrected(x) = (1 - lambda(x)) * y_global(x) + lambda(x) * y_local(x)

``y_local`` is computed from the query's nearest same-endpoint training neighbours (Tanimoto
similarity on Morgan fingerprints); ``lambda(x)`` in [0, 1] decreases with neighbour distance,
with between-neighbour disagreement, and with neighbour interval width, and is zero when no
neighbour clears the similarity threshold (then the global prediction is returned unchanged).

Three estimators for ``y_local``:

``neighbour``
    The similarity-weighted mean of the neighbours' measured values (a point estimate).
``kernel``
    The value minimising the weighted soft-threshold distance to the neighbours' credible
    intervals, under similarity**power weights. The objective is piecewise linear in the
    candidate, so the optimum sits at an interval bound; of the bounds attaining it, the one
    nearest the global prediction is returned.
``residual``
    The global prediction plus the similarity**power-weighted mean of the neighbours'
    residuals (measured minus the global model's out-of-fold prediction at each neighbour).
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
    anchor: float,
) -> float:
    """The value minimising the weighted soft-threshold distance to a set of intervals.

    The objective ``sum_i w_i * (max(0, y - high_i) + max(0, low_i - y))`` is convex and
    piecewise linear in ``y``, so a minimiser lies at one of the bounds. The optimum is often a
    flat interval (e.g. a single reference ``[0, 10]`` is matched by any value inside it); of
    the points that achieve it, the one closest to ``anchor`` (the global prediction) is
    returned, so a global prediction already inside the admissible region is left where it is
    rather than dragged to an arbitrary bound.
    """
    candidates = np.concatenate([low, high])

    def objective(y: float) -> float:
        above = np.clip(y - high, a_min=0.0, a_max=None)
        below = np.clip(low - y, a_min=0.0, a_max=None)
        return float((weights * (above + below)).sum())

    best = min(objective(float(c)) for c in candidates)
    # The anchor is itself admissible when it does no worse than the bound optimum (it lies in
    # the flat region); otherwise take the optimal bound nearest the anchor.
    if objective(anchor) <= best + 1e-9:
        return float(anchor)
    optimal = [float(c) for c in candidates if objective(float(c)) <= best + 1e-9]
    return min(optimal, key=lambda c: abs(c - anchor))


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
    #: Mean neighbour interval width (in pIC50) at which the width factor ``exp(-width / scale)``
    #: falls to ``1/e``. Independent of ``disagreement_scale``: a single neighbour has zero
    #: disagreement but can still be wide.
    width_scale: float = 2.0
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
        order = eligible[np.argsort(-similarities[eligible])][: self.max_neighbours]
        similarity = similarities[order]
        # The trust weights (for lambda) are always the sharp kernel; the estimator weights
        # differ: "neighbour" is a plain similarity-weighted point, "kernel" leans harder on
        # the closest match via similarity**power, so the two are genuinely distinct.
        trust_weight = similarity**self.power
        total = float(trust_weight.sum())
        if total <= 0:
            return global_value

        value = reference["value"][order]
        lower = reference["lower"][order]
        upper = reference["upper"][order]
        residual = reference["residual"][order]

        if self.estimator == "residual":
            local = global_value + float((trust_weight * residual).sum() / total)
            spread_basis = residual
        elif self.estimator == "kernel":
            # Interval-aware: the point minimising the weighted soft-threshold distance to the
            # neighbours' credible intervals under the sharp kernel weights.
            local = _interval_point(lower, upper, similarity**self.power, global_value)
            spread_basis = value
        else:  # "neighbour": the plain similarity-weighted mean of the neighbour values
            local = float((similarity * value).sum() / float(similarity.sum()))
            spread_basis = value

        # lambda = (W / (W + shrinkage)) * exp(-disagreement / s_d) * exp(-mean_width / s_w),
        # where W is the summed neighbour weight, disagreement the weighted sd of spread_basis,
        # and mean_width the weighted mean interval width.
        mean = float((trust_weight * spread_basis).sum() / total)
        variance = float((trust_weight * (spread_basis - mean) ** 2).sum() / total)
        disagreement = float(np.sqrt(max(variance, 0.0)))
        mean_width = float((trust_weight * (upper - lower)).sum() / total)
        lam = (
            (total / (total + self.shrinkage))
            * float(np.exp(-disagreement / self.disagreement_scale))
            * float(np.exp(-mean_width / self.width_scale))
        )
        return (1.0 - lam) * global_value + lam * local
