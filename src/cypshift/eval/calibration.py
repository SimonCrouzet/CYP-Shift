"""Decompose R² into ranking, spread and offset, to separate what recalibration can fix.

Under squared error the coefficient of determination factors exactly as

    R² = ρ² − (s − ρ)² − b²

with ρ the Pearson correlation between prediction and truth, s = σ_pred / σ_true the
spread ratio, and b = (mean_pred − mean_true) / σ_true the offset in units of the truth's
spread. The identity attributes every point of lost R² to one of three causes: a ranking
ceiling below one (R² can never exceed ρ²), a predicted spread away from the R²-optimal
ρ·σ_true, or a mean offset. The ranking is invariant to the affine map that fixes the last
two, so the split says plainly which part of a change a model earned and which a
recalibration could have supplied. The decomposition is standard; it earns its place here
because the soft-threshold metric makes ranking and calibration pull in different
directions, and a single R² hides that.
"""

import numpy as np
from numpy.typing import ArrayLike


def r2_decomposition(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Split R² into its ranking ceiling and the spread and offset it loses.

    Returns ``rho`` (Pearson correlation), ``spread_ratio`` ``s`` and ``offset`` ``b``,
    the reconstructed ``r2`` (equal to ``sklearn``'s on finite pairs), and the three
    additive terms ``ranking_ceiling`` (ρ²), ``spread_loss`` ((s − ρ)²) and ``offset_loss``
    (b²), so that ``r2 = ranking_ceiling − spread_loss − offset_loss``.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    sigma_y = float(y.std())
    sigma_p = float(p.std())
    if sigma_y == 0.0:
        raise ValueError("truth has zero spread; R² is undefined")
    rho = float(np.corrcoef(y, p)[0, 1]) if sigma_p > 0.0 else 0.0
    spread_ratio = sigma_p / sigma_y
    offset = (float(p.mean()) - float(y.mean())) / sigma_y
    ranking_ceiling = rho**2
    spread_loss = (spread_ratio - rho) ** 2
    offset_loss = offset**2
    return {
        "rho": rho,
        "spread_ratio": spread_ratio,
        "offset": offset,
        "r2": ranking_ceiling - spread_loss - offset_loss,
        "ranking_ceiling": ranking_ceiling,
        "spread_loss": spread_loss,
        "offset_loss": offset_loss,
    }
