"""Choosing the point to submit, given a predictive distribution.

The scored loss has a dead zone, so the point that minimises expected error is not
the posterior mean. Writing ``U = y + w_high`` and ``L = y - w_low`` for the credible
bounds implied by a possible true value ``y``, the expected loss at a candidate
prediction ``p`` is

.. math::

    \\mathbb{E}\\left[(p - U)_+ + (L - p)_+\\right]

which is convex and piecewise linear in ``p``. Its derivative is
``P(U < p) - P(L > p)``, so the minimiser is the point where the probability of
overshooting every upper bound first exceeds the probability of undershooting every
lower bound. That is a weighted-quantile problem with an exact answer; no grid search
is needed, and :func:`expected_soft_loss` exists so the exact solution can be checked
against a brute-force scan in the tests.

Why this matters here. Under a bimodal active/inactive posterior the mean lands in
the empty middle and takes error from both sides, while the correct decision commits
to one mode. On the public leaderboard the team with the best mean absolute error of
all entrants places seventh, which is the same effect seen from the outside.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from cypshift.decision.band import BandWidthModel
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)


def expected_soft_loss(
    candidates: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean dead-zone loss at each candidate prediction, over drawn credible bounds.

    Args:
        candidates: Candidate predictions, shape ``(n_candidates,)``.
        lower_bounds: Drawn lower bounds, shape ``(n_draws,)``.
        upper_bounds: Drawn upper bounds, shape ``(n_draws,)``.

    Returns:
        Expected loss per candidate, shape ``(n_candidates,)``.
    """
    grid = candidates[:, None]
    above = np.clip(grid - upper_bounds[None, :], 0, None)
    below = np.clip(lower_bounds[None, :] - grid, 0, None)
    return (above + below).mean(axis=1)


def optimal_placement(
    lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
) -> float:
    """The exact minimiser of the expected dead-zone loss.

    The loss is convex and piecewise linear with kinks only at the drawn bounds, so
    the minimum is attained at one of them. The derivative condition
    ``P(U < p) >= P(L > p)`` locates it in ``O(n log n)``.
    """
    upper_sorted = np.sort(upper_bounds)
    lower_sorted = np.sort(lower_bounds)
    n_draws = len(upper_sorted)
    candidates = np.union1d(lower_sorted, upper_sorted)

    n_overshooting = np.searchsorted(upper_sorted, candidates, side="right")
    n_undershooting = n_draws - np.searchsorted(lower_sorted, candidates, side="left")
    crossed = np.flatnonzero(n_overshooting >= n_undershooting)
    return float(candidates[crossed[0]] if len(crossed) else candidates[-1])


def place(
    distribution: PredictiveDistribution,
    band_models: dict[str, BandWidthModel],
    n_draws: int = 400,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Turn a predictive distribution into the points to submit.

    Args:
        distribution: What a route predicted.
        band_models: One fitted :class:`BandWidthModel` per endpoint.
        n_draws: Monte Carlo draws per compound.
        seed: Fixes the draws.

    Returns:
        Shape ``(n_compounds, n_endpoints)``, the prediction to submit.

    Raises:
        KeyError: If an endpoint has no band model.
    """
    missing = set(distribution.endpoints) - set(band_models)
    if missing:
        raise KeyError(f"no band model for {sorted(missing)}")

    draws = distribution.draw(n_draws, seed=seed)
    placements = np.empty((distribution.n_compounds, len(distribution.endpoints)))
    for column, endpoint in enumerate(distribution.endpoints):
        values = draws[:, :, column]
        lower_half, upper_half = band_models[endpoint].predict(values)
        lower_bounds, upper_bounds = values - lower_half, values + upper_half
        for compound in range(distribution.n_compounds):
            placements[compound, column] = optimal_placement(
                lower_bounds[:, compound], upper_bounds[:, compound]
            )
    logger.info("placed %d compounds across %d endpoints", *placements.shape)
    return placements
