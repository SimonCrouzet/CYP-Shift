"""Choosing the point to submit, given a predictive distribution.

The scored loss has a soft threshold, so the point that minimises expected error is not
the posterior mean. Writing ``U = y + w_high`` and ``L = y - w_low`` for the credible
bounds implied by a possible true value ``y``, the expected loss at a candidate
prediction ``p`` is

.. math::

    \\mathbb{E}\\left[(p - U)_+ + (L - p)_+\\right]

which is convex and piecewise linear in ``p``. Its derivative is
``P(U < p) - P(L > p)``, so the minimiser is the point where the probability of
overshooting every upper bound first exceeds the probability of undershooting every
lower bound. That is a weighted-quantile problem with an exact answer; no grid search
is needed, and :func:`expected_loss` exists so the exact solution can be checked
against a brute-force scan in the tests.

Why this matters here. Under a bimodal active/inactive posterior the mean lands in
the empty middle and takes error from both sides, while the correct decision commits
to one mode. On the public leaderboard the team with the best mean absolute error of
all entrants places seventh, which is the same effect seen from the outside.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from cypshift.decision.interval_width import IntervalWidthModel
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)


def expected_loss(
    candidates: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean soft-threshold loss at each candidate prediction, over drawn credible bounds.

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


def loss_minimising_estimate(
    lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
) -> float:
    """The exact minimiser of the expected soft-threshold loss.

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


def choose_estimates(
    distribution: PredictiveDistribution,
    width_models: dict[str, IntervalWidthModel],
    n_draws: int = 400,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Turn a predictive distribution into the points to submit.

    Args:
        distribution: What a model predicted.
        width_models: One fitted :class:`IntervalWidthModel` per endpoint.
        n_draws: Monte Carlo draws per compound.
        seed: Fixes the draws.

    Returns:
        Shape ``(n_compounds, n_endpoints)``, the prediction to submit.

    Raises:
        KeyError: If an endpoint has no interval model.
    """
    missing = set(distribution.endpoints) - set(width_models)
    if missing:
        raise KeyError(f"no interval model for {sorted(missing)}")

    draws = distribution.draw(n_draws, seed=seed)
    point_estimates = np.empty((distribution.n_compounds, len(distribution.endpoints)))
    for column, endpoint in enumerate(distribution.endpoints):
        values = draws[:, :, column]
        lower_half, upper_half = width_models[endpoint].predict(values)
        lower_bounds, upper_bounds = values - lower_half, values + upper_half
        for compound in range(distribution.n_compounds):
            point_estimates[compound, column] = loss_minimising_estimate(
                lower_bounds[:, compound], upper_bounds[:, compound]
            )
    logger.info("chosen %d compounds across %d endpoints", *point_estimates.shape)
    return point_estimates
