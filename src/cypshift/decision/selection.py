"""Deciding, per endpoint, whether to place at all.

Placement is not uniformly good. Measured on the diverse ensemble with a paired
bootstrap, it beats the posterior mean on CYP3A4 in 99% of samples and on CYP2D6 in
84%, while losing on CYP1A2 and CYP2C9 in about 77% and 83%. Averaging those four
into one macro number hides two clear wins behind two probable losses and leaves a
difference that is not resolvable.

Its value tracks how informative the posterior is for that endpoint, which varies a
great deal: cross-validated R-squared runs from 0.49 on CYP3A4 to 0.14 on CYP2D6.
Where the model is confident and wrong, committing to a mode commits to a wrong one.

So whether to place is itself a decision, and it is made here from held-out evidence
rather than assumed. The choice is fitted on calibration data and applied to
evaluation data, never selected on the same compounds it is then scored against.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.eval.scoring import paired_difference

logger = logging.getLogger(__name__)

#: A placed endpoint must win this share of paired bootstrap samples. Set above a
#: coin flip by a clear margin, because reverting to the posterior mean costs nothing
#: while a wrongly placed endpoint costs real score.
DEFAULT_WIN_RATE_THRESHOLD = 0.60


@dataclass
class PlacementSelector:
    """Which endpoints benefit from placement, decided on held-out data."""

    place_endpoint: dict[str, bool] = field(default_factory=dict)
    win_rates: dict[str, float] = field(default_factory=dict)
    threshold: float = DEFAULT_WIN_RATE_THRESHOLD

    @classmethod
    def fit(
        cls,
        placed: NDArray[np.float64],
        mean: NDArray[np.float64],
        truth: pd.DataFrame,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        key: str = "Molecule_Name",
        threshold: float = DEFAULT_WIN_RATE_THRESHOLD,
    ) -> "PlacementSelector":
        """Compare placed against mean predictions on calibration compounds.

        Args:
            placed: Placed predictions, shape ``(n_compounds, n_endpoints)``.
            mean: Posterior means, same shape.
            truth: Calibration ground truth with credible bounds and ``key``.
            endpoints: Endpoint names matching the prediction columns.
            key: Compound identifier column.
            threshold: Share of paired bootstrap samples a placed endpoint must win.

        Returns:
            A selector recording the decision and the evidence behind it.
        """
        frames = {}
        for label, values in (("placed", placed), ("mean", mean)):
            frame = truth[[key]].copy()
            for column, endpoint in enumerate(endpoints):
                frame[endpoint] = values[:, column]
            frames[label] = frame

        comparison = paired_difference(frames["placed"], frames["mean"], truth, key=key)
        win_rates = {e: float(comparison.loc[e, "win_rate"]) for e in endpoints}
        decisions = {e: rate >= threshold for e, rate in win_rates.items()}
        for endpoint, rate in win_rates.items():
            logger.info(
                "%s: placement wins %.1f%% of paired samples -> %s",
                endpoint,
                100 * rate,
                "place" if decisions[endpoint] else "use the mean",
            )
        return cls(decisions, win_rates, threshold)

    def apply(
        self,
        placed: NDArray[np.float64],
        mean: NDArray[np.float64],
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
    ) -> NDArray[np.float64]:
        """Take placed predictions where they were shown to help, means elsewhere."""
        missing = set(endpoints) - set(self.place_endpoint)
        if missing:
            raise KeyError(f"no decision recorded for {sorted(missing)}")
        chosen = mean.copy()
        for column, endpoint in enumerate(endpoints):
            if self.place_endpoint[endpoint]:
                chosen[:, column] = placed[:, column]
        return chosen
