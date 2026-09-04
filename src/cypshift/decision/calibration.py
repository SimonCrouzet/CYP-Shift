"""Correcting the size of a predictive distribution without disturbing its shape.

A seeded ensemble ranks errors weakly but states their size very badly. Measured on
this dataset, five random forests differing only in seed agree with one another to
within 0.03 log units while being wrong by 0.45 -- an understatement of twelve to
fifteen times, with 90% intervals covering 4-10% of compounds instead of 90%. Seed
disagreement captures almost none of the real uncertainty, because every member saw
the same data and shares the same failure modes.

That matters here because the placement layer optimises against whatever posterior it
is handed. Given one that is effectively a point mass it cannot do the job it exists
for, and where the model is confident and wrong it will commit harder to the wrong
answer.

Split conformal prediction fixes the size while keeping the ordering. The residual is
divided by the predicted deviation *before* the quantile is taken, so the spread
still decides which compounds get wider intervals and conformal decides how wide they
all are. Without that division every compound would receive the same width: correct
on average and uninformative per compound.

The correction is applied as an added aleatoric term rather than by stretching the
samples, so a route's predicted shape -- including genuine multi-modality, where it
has any -- survives calibration.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)

#: Floor on the predicted deviation before dividing, so a near-zero spread cannot
#: produce an unbounded conformal score.
DEFAULT_STD_FLOOR = 0.05


@dataclass
class ConformalCalibrator:
    """Per-endpoint multipliers taking a predicted spread to its honest size."""

    multipliers: dict[str, float] = field(default_factory=dict)
    level: float = 0.90
    std_floor: float = DEFAULT_STD_FLOOR

    @classmethod
    def fit(
        cls,
        distribution: PredictiveDistribution,
        y_true: NDArray[np.float64],
        level: float = 0.90,
        std_floor: float = DEFAULT_STD_FLOOR,
    ) -> "ConformalCalibrator":
        """Learn a multiplier per endpoint on data the route did not train on.

        Args:
            distribution: Predictions on the calibration compounds.
            y_true: Their measured values, shape ``(n_compounds, n_endpoints)``, NaN
                where unmeasured.
            level: Nominal coverage the calibrated interval should achieve.
            std_floor: Lower bound on the predicted deviation before dividing.

        Raises:
            ValueError: If an endpoint has no measured calibration compound.
        """
        mean, std = distribution.mean, distribution.total_std
        multipliers = {}
        for column, endpoint in enumerate(distribution.endpoints):
            measured = np.isfinite(y_true[:, column])
            if not measured.any():
                raise ValueError(f"{endpoint}: no measured compounds to calibrate on")
            residual = np.abs(y_true[measured, column] - mean[measured, column])
            scores = residual / np.maximum(std[measured, column], std_floor)
            n = len(scores)
            rank = min(int(np.ceil((n + 1) * level)), n)
            multipliers[endpoint] = float(np.sort(scores)[rank - 1])
            logger.info(
                "%s: conformal multiplier %.1f from %d compounds",
                endpoint,
                multipliers[endpoint],
                n,
            )
        return cls(multipliers, level, std_floor)

    def apply(self, distribution: PredictiveDistribution) -> PredictiveDistribution:
        """Return a copy whose total spread is the calibrated one.

        The shortfall is added as aleatoric variance, so the route's own sample shape
        is preserved and only the width changes.
        """
        missing = set(distribution.endpoints) - set(self.multipliers)
        if missing:
            raise KeyError(f"no multiplier for {sorted(missing)}")

        multiplier = np.array([self.multipliers[e] for e in distribution.endpoints])
        target_variance = (np.maximum(distribution.total_std, self.std_floor) * multiplier) ** 2
        shortfall = np.clip(target_variance - distribution.epistemic_variance, 0.0, None)
        added = np.broadcast_to(shortfall, distribution.samples.shape).copy()
        if distribution.aleatoric_variance is not None:
            added = added + distribution.aleatoric_variance
        return PredictiveDistribution(distribution.samples, distribution.endpoints, added)
