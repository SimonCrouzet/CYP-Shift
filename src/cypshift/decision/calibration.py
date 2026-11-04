"""Correcting the size of a predictive distribution without disturbing its shape.

A seeded ensemble ranks errors weakly but states their size very badly. Measured on
this dataset, five random forests differing only in seed agree with one another to
within 0.03 log units while being wrong by 0.45 -- an understatement of twelve to
fifteen times, with 90% intervals covering 4-10% of compounds instead of 90%. Seed
disagreement captures almost none of the real uncertainty, because every member saw
the same data and shares the same failure modes.

That matters here because the point-estimate step optimises against whatever posterior it
is handed. Given one that is effectively a point mass it cannot do the job it exists
for, and where the model is confident and wrong it will commit harder to the wrong
answer.

Split conformal prediction fixes the size while keeping the ordering. The residual is
divided by the predicted deviation *before* the quantile is taken, so the spread
still decides which compounds get wider intervals and conformal decides how wide they
all are. Without that division every compound would receive the same width: correct
on average and uninformative per compound.

The correction is applied as an added aleatoric term rather than by stretching the
samples, so a model's predicted shape -- including genuine multi-modality, where it
has any -- survives calibration.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)

#: The floor guarding against division by a near-zero spread is set as a low quantile
#: of the observed spread rather than as an absolute constant. A fixed floor clamps
#: every compound whenever a model is confident, which flattens the spread to a
#: constant and destroys the per-compound ranking conformal is meant to preserve.
DEFAULT_FLOOR_QUANTILE = 0.05


@dataclass
class ConformalCalibrator:
    """Per-endpoint multipliers taking a predicted spread to its honest size."""

    multipliers: dict[str, float] = field(default_factory=dict)
    floors: dict[str, float] = field(default_factory=dict)
    level: float = 0.90

    @classmethod
    def fit(
        cls,
        distribution: PredictiveDistribution,
        y_true: NDArray[np.float64],
        level: float = 0.90,
        floor_quantile: float = DEFAULT_FLOOR_QUANTILE,
    ) -> "ConformalCalibrator":
        """Learn a multiplier per endpoint on data the model did not train on.

        Args:
            distribution: Predictions on the calibration compounds.
            y_true: Their measured values, shape ``(n_compounds, n_endpoints)``, NaN
                where unmeasured.
            level: Nominal coverage the calibrated interval should achieve.
            floor_quantile: Quantile of the observed spread used as the division floor.

        Raises:
            ValueError: If an endpoint has no measured calibration compound.
        """
        mean, std = distribution.mean, distribution.total_std
        multipliers, floors = {}, {}
        for column, endpoint in enumerate(distribution.endpoints):
            measured = np.isfinite(y_true[:, column])
            if not measured.any():
                raise ValueError(f"{endpoint}: no measured compounds to calibrate on")
            observed_std = std[measured, column]
            floor = float(max(np.quantile(observed_std, floor_quantile), 1e-6))
            residual = np.abs(y_true[measured, column] - mean[measured, column])
            scores = residual / np.maximum(observed_std, floor)
            n = len(scores)
            rank = min(int(np.ceil((n + 1) * level)), n)
            multipliers[endpoint] = float(np.sort(scores)[rank - 1])
            floors[endpoint] = floor
            logger.info(
                "%s: conformal multiplier %.1f, floor %.3f, from %d compounds",
                endpoint,
                multipliers[endpoint],
                floor,
                n,
            )
        return cls(multipliers, floors, level)

    def apply(self, distribution: PredictiveDistribution) -> PredictiveDistribution:
        """Return a copy whose total spread is the calibrated one.

        The conformal quantile is an interval *half-width* at the nominal level, not a
        standard deviation, so it is divided by the corresponding normal quantile
        before being used as one. Skipping that conversion inflates the interval by
        the same factor and overshoots coverage.

        The aleatoric variance is *set* to the gap between the calibrated target and the
        model's own (epistemic) sample spread, so the total reaches the target. It must not
        be added on top of the distribution's existing aleatoric variance: the target is
        derived from the total spread, which already reflects that term, so re-adding it
        double-counts (asking for total variance 1 on a distribution already carrying 1
        would otherwise return 2).
        """
        missing = set(distribution.endpoints) - set(self.multipliers)
        if missing:
            raise KeyError(f"no multiplier for {sorted(missing)}")

        multiplier = np.array([self.multipliers[e] for e in distribution.endpoints])
        floor = np.array([self.floors[e] for e in distribution.endpoints])
        z = norm.ppf(0.5 + self.level / 2)
        half_width = np.maximum(distribution.total_std, floor) * multiplier
        target_variance = (half_width / z) ** 2
        aleatoric = np.clip(target_variance - distribution.epistemic_variance, 0.0, None)
        added = np.broadcast_to(aleatoric, distribution.samples.shape).copy()
        return PredictiveDistribution(distribution.samples, distribution.endpoints, added)
