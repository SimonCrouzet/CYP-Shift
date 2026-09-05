"""Disagreement between models as an estimate of epistemic uncertainty.

Where independently-built models disagree about a compound, at least one is wrong, and
the size of that disagreement is a per-compound signal of how much to trust the
prediction. This is the standard reading of ensemble disagreement as *epistemic*
uncertainty -- uncertainty from the model rather than from measurement noise.

An ensemble's *internal* spread underestimates it. Members that share an architecture,
a featurisation and a training set disagree about sampling noise and agree about
everything their shared assumptions get wrong. Two different model families disagree
about the assumptions too, and that extra disagreement is the part that tracks being
wrong for structural reasons -- a compound unlike anything in training, or one whose
activity runs through a mechanism the features do not encode.

What this does *not* do is change the submitted number by itself: the metric scores a
point estimate against the organisers' own credible interval, so a wide or narrow
uncertainty is not something a submission carries. Its uses are the ones that act on an
uncertainty estimate -- flagging low-confidence predictions, prioritising where more
data would pay, and feeding a per-compound weight to whatever consumes it downstream.

Disagreement is an unscaled signal; :func:`calibrate_width` fits the one scalar that
turns it into an interval of a requested coverage, for reporting. Whether it earns its
place is a tuning question, not one to settle from a single reduced experiment.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Disagreement:
    """Per-compound spread across model families, and the scale that calibrates it."""

    spread: NDArray[np.float64]
    scale: float
    floor: float

    def interval(
        self, centre: NDArray[np.float64], spread: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], ...]:
        """Lower and upper bounds around a point estimate.

        Args:
            centre: The point estimates to bound.
            spread: Cross-model disagreement for these compounds. Defaults to the
                calibration-set spread, which is correct only when ``centre`` is that
                same set; pass the new compounds' spread to apply the fitted scale to
                them.
        """
        source = self.spread if spread is None else np.asarray(spread, dtype=float)
        half = np.maximum(source * self.scale, self.floor)
        return centre - half, centre + half


def cross_model_spread(predictions: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Standard deviation across model families, per compound and endpoint.

    Args:
        predictions: One array per model family, all the same shape.

    Raises:
        ValueError: If fewer than two families are given, or their shapes differ.
            A spread over one model is zero everywhere, which would silently produce
            an interval of the floor width and look like confidence.
    """
    if len(predictions) < 2:
        raise ValueError("disagreement needs at least two model families")
    shapes = {name: np.asarray(v).shape for name, v in predictions.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"model predictions differ in shape: {shapes}")
    stacked = np.stack([np.asarray(v, dtype=float) for v in predictions.values()])
    spread = stacked.std(axis=0, ddof=1 if len(predictions) > 1 else 0)
    logger.info(
        "cross-model spread over %d families: median %.3f, 90th percentile %.3f",
        len(predictions),
        float(np.median(spread)),
        float(np.percentile(spread, 90)),
    )
    return spread


def calibrate_width(
    spread: NDArray[np.float64],
    residual: NDArray[np.float64],
    level: float = 0.95,
    floor_quantile: float = 0.1,
) -> Disagreement:
    """Fit the scalar that turns raw disagreement into an interval of the given coverage.

    Conformal in spirit: the scale is the quantile of ``|residual| / spread`` that
    leaves the requested fraction of held-out compounds inside. Two details were
    learned the hard way in this project:

    * The quantile is used directly, not treated as a standard deviation. Multiplying
      it again by the normal quantile overshot coverage to 94-100%.
    * The floor is a quantile of the observed spread rather than a fixed constant. A
      constant of 0.05 clamped a distribution whose median spread was 0.03, flattening
      every interval to the same width.

    Args:
        spread: Cross-model disagreement on a calibration set.
        residual: Signed or unsigned error of the point estimate on the same set.
        level: Target coverage.
        floor_quantile: Minimum half-width, as a quantile of the observed spread.

    Returns:
        The fitted :class:`Disagreement`, carrying the calibration-set spread.
    """
    spread = np.asarray(spread, dtype=float).ravel()
    residual = np.abs(np.asarray(residual, dtype=float)).ravel()
    if spread.shape != residual.shape:
        raise ValueError(f"spread {spread.shape} and residual {residual.shape} differ")

    floor = float(np.quantile(spread[spread > 0], floor_quantile)) if (spread > 0).any() else 0.0
    safe = np.maximum(spread, max(floor, 1e-9))
    scale = float(np.quantile(residual / safe, level))
    logger.info(
        "calibrated disagreement: scale %.3f, floor %.4f, target coverage %.0f%%",
        scale,
        floor,
        level * 100,
    )
    return Disagreement(spread=spread, scale=scale, floor=floor)


def gaussian_width(sigma: NDArray[np.float64], level: float = 0.95) -> NDArray[np.float64]:
    """Half-width of a Gaussian interval, for comparison against the calibrated one.

    The naive alternative: treat the spread as a standard deviation and multiply by the
    normal quantile. Included so the two can be scored against each other rather than
    assumed.
    """
    return np.asarray(sigma, dtype=float) * float(norm.ppf(0.5 + level / 2))
