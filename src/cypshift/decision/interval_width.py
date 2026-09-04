"""Predicting how wide a compound's credible interval will be.

The scored metric forgives any prediction landing inside the fitted dose-response
curve's 95% credible interval, so choosing a prediction well requires knowing how
wide that interval is likely to be. Width is strongly activity-dependent: on the
training set the median is 0.27-0.53 log units above pIC50 4 and 1.21-2.59 below it,
because a curve with no measurable inflection is barely constrained.

The dependence is estimated non-parametrically, as a binned median with linear
interpolation. It is not assumed monotone: width falls steeply across the assay floor
and then flattens, which a monotone fit would smear.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class IntervalWidthModel:
    """Half-widths of the credible interval as a function of the true value.

    Attributes:
        levels: Bin centres in pIC50, ascending.
        lower_half_width: Median ``y - conf_low`` at each level.
        upper_half_width: Median ``conf_high - y`` at each level.
    """

    levels: NDArray[np.float64]
    lower_half_width: NDArray[np.float64]
    upper_half_width: NDArray[np.float64]

    @classmethod
    def fit(
        cls,
        y_true: NDArray[np.float64],
        y_true_lower: NDArray[np.float64],
        y_true_upper: NDArray[np.float64],
        n_bins: int = 12,
    ) -> "IntervalWidthModel":
        """Estimate half-widths by binning observed intervals on activity.

        Args:
            y_true: Measured pIC50 values.
            y_true_lower: Their lower credible bounds.
            y_true_upper: Their upper credible bounds.
            n_bins: Quantile bins across the observed activity range.

        Raises:
            ValueError: If fewer than ``n_bins`` complete observations are supplied.
        """
        complete = np.isfinite(y_true) & np.isfinite(y_true_lower) & np.isfinite(y_true_upper)
        y, low, high = y_true[complete], y_true_lower[complete], y_true_upper[complete]
        if len(y) < n_bins:
            raise ValueError(f"need at least {n_bins} complete observations, got {len(y)}")

        edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
        assignment = np.clip(np.digitize(y, edges[1:-1]), 0, len(edges) - 2)
        levels, lower, upper = [], [], []
        for index in range(len(edges) - 1):
            in_bin = assignment == index
            if not in_bin.any():
                continue
            levels.append(float(np.median(y[in_bin])))
            lower.append(float(np.median(y[in_bin] - low[in_bin])))
            upper.append(float(np.median(high[in_bin] - y[in_bin])))
        logger.info(
            "interval width across %.2f-%.2f pIC50: lower %.2f-%.2f, upper %.2f-%.2f",
            levels[0],
            levels[-1],
            min(lower),
            max(lower),
            min(upper),
            max(upper),
        )
        return cls(np.array(levels), np.array(lower), np.array(upper))

    def predict(self, y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolated ``(lower_half_width, upper_half_width)`` at each value.

        Values outside the fitted range take the nearest bin's width rather than
        extrapolating, since a linear extrapolation of an interval width can go
        negative.
        """
        return (
            np.interp(y, self.levels, self.lower_half_width),
            np.interp(y, self.levels, self.upper_half_width),
        )
