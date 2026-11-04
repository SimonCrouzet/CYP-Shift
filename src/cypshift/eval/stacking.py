"""A small penalised correction fitted on out-of-fold predictions against the interval loss.

For one endpoint the correction is ``base + sum_j w_j * z_j``: ``base`` is the incumbent's
prediction, ``z_j`` are standardised extra columns (other members' predictions minus the
incumbent's, auxiliary-head outputs), and the weights minimise the summed hinge
``max(0, low - p) + max(0, p - high)`` plus ``l2 * sum(w ** 2)``. The weights are free to be
negative. The penalty is chosen by leaving one fold of the training compounds out at a time, and
:func:`cross_validated_stack` scores the whole procedure on compounds whose fold the weights never
saw, so its predictions are the honest estimate of what the correction does to new compounds.

The columns come from out-of-fold predictions, so a weight fitted here applies to predictions of
the same kind (a model's predictions of compounds it was not trained on).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

#: Penalties tried, per standard deviation of a standardised column.
L2_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def hinge(
    prediction: NDArray[np.float64], low: NDArray[np.float64], high: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distance of each prediction outside its interval; zero inside."""
    return np.clip(low - prediction, 0.0, None) + np.clip(prediction - high, 0.0, None)


@dataclass(frozen=True)
class Stack:
    """A fitted correction: the weight of each standardised column, the penalty it was fitted
    with, and the centre and scale that standardise a column."""

    weights: NDArray[np.float64]
    l2: float
    centre: NDArray[np.float64]
    scale: NDArray[np.float64]


def standardise(
    columns: NDArray[np.float64], centre: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Columns minus their centre over their scale; a missing value becomes the centre (zero)."""
    return np.nan_to_num((columns - centre) / scale, nan=0.0)


def fit_weights(
    z: NDArray[np.float64],
    base: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    l2: float,
) -> NDArray[np.float64]:
    """The weights minimising mean hinge of ``base + z @ w`` plus ``l2 * sum(w ** 2)``."""
    if z.shape[1] == 0:
        return np.zeros(0)

    def objective(weights: NDArray[np.float64]) -> float:
        return float(hinge(base + z @ weights, low, high).mean() + l2 * np.sum(weights**2))

    return minimize(
        objective,
        np.zeros(z.shape[1]),
        method="Powell",
        options={"maxiter": 20000, "xtol": 1e-6, "ftol": 1e-8},
    ).x


def choose_l2(
    z: NDArray[np.float64],
    base: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    folds: NDArray[np.int_],
    grid: tuple[float, ...] = L2_GRID,
) -> float:
    """The penalty of ``grid`` with the lowest hinge on the compounds of each fold, summed over the
    folds, when the weights are fitted on the other folds."""
    totals = []
    for l2 in grid:
        total = 0.0
        for fold in np.unique(folds):
            held = folds == fold
            weights = fit_weights(z[~held], base[~held], low[~held], high[~held], l2)
            total += float(hinge(base[held] + z[held] @ weights, low[held], high[held]).sum())
        totals.append(total)
    return grid[int(np.argmin(totals))]


def fit_stack(
    base: NDArray[np.float64],
    columns: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    folds: NDArray[np.int_],
    grid: tuple[float, ...] = L2_GRID,
) -> Stack:
    """Standardise the columns, choose the penalty by leave-one-fold-out, and fit the weights on
    all the compounds."""
    centre = np.nanmean(columns, axis=0)
    scale = np.nanstd(columns, axis=0) + 1e-9
    z = standardise(columns, centre, scale)
    l2 = choose_l2(z, base, low, high, folds, grid)
    return Stack(fit_weights(z, base, low, high, l2), l2, centre, scale)


def apply_stack(
    stack: Stack, base: NDArray[np.float64], columns: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The corrected predictions of ``base`` given its columns."""
    return base + standardise(columns, stack.centre, stack.scale) @ stack.weights


def cross_validated_stack(
    base: NDArray[np.float64],
    columns: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
    folds: NDArray[np.int_],
    grid: tuple[float, ...] = L2_GRID,
) -> tuple[NDArray[np.float64], list[float]]:
    """Corrected predictions of every compound from weights fitted without its fold, and the
    penalty chosen for each fold."""
    corrected = np.empty_like(base)
    chosen = []
    for fold in np.unique(folds):
        held = folds == fold
        stack = fit_stack(base[~held], columns[~held], low[~held], high[~held], folds[~held], grid)
        corrected[held] = apply_stack(stack, base[held], columns[held])
        chosen.append(stack.l2)
    return corrected, chosen


def restore_moments(
    raw: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``raw`` rescaled linearly to the mean and standard deviation of ``reference``: the order of
    the compounds is kept and the level and spread are the reference's."""
    spread = raw.std()
    if spread < 1e-9:
        return np.full_like(raw, reference.mean())
    return (raw - raw.mean()) / spread * reference.std() + reference.mean()
