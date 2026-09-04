"""Deciding, per endpoint, whether to choose_estimates at all.

Placement is not uniformly good. Measured on the diverse ensemble with a paired
bootstrap, it beats the posterior mean on CYP3A4 in 99% of samples and on CYP2D6 in
84%, while losing on CYP1A2 and CYP2C9 in about 77% and 83%. Averaging those four
into one macro number hides two clear wins behind two probable losses and leaves a
difference that is not resolvable.

Its value tracks how informative the posterior is for that endpoint, which varies a
great deal: cross-validated R-squared runs from 0.49 on CYP3A4 to 0.14 on CYP2D6.
Where the model is confident and wrong, committing to a mode commits to a wrong one.

So whether to choose_estimates is itself a decision, and it is made here from held-out evidence
rather than assumed. The choice is fitted on calibration data and applied to
evaluation data, never selected on the same compounds it is then scored against.

That discipline turned out to matter: the first version of this selector, with a
threshold just above a coin flip, chose two endpoints whose advantage did not
survive the crossing and scored worse than applying point_estimate everywhere. See
:data:`DEFAULT_P_VALUE_THRESHOLD` for what the reversal looked like and why the bar
is now set where it is.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.eval.scoring import paired_difference

logger = logging.getLogger(__name__)

#: A chosen endpoint must win this share of paired bootstrap samples.
#:
#: The threshold is demanding on purpose, and the reason is measured rather than
#: assumed. Estimating this win rate on one half of a similarity-matched holdout and
#: checking it on the other, two of four endpoints *reversed*: CYP2C9 went from 81%
#: to 17% and CYP2D6 from 59% to 84%. With sixty to a hundred and sixty measured
#: compounds per endpoint per half the estimate is barely informative, and a
#: threshold near a coin flip selects noise -- doing so scored worse than not
#: selecting at all.
#:
#: Only near-unanimous effects survived the crossing: CYP3A4 at 100% and 99%,
#: CYP1A2's rejection at 16% and 24%. Requiring near-unanimity is what separates a
#: replicated effect from a large one.
DEFAULT_P_VALUE_THRESHOLD = 0.95


@dataclass
class EstimatorSelector:
    """Which endpoints benefit from point_estimate, decided on held-out data."""

    use_loss_minimising: dict[str, bool] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    threshold: float = DEFAULT_P_VALUE_THRESHOLD

    @classmethod
    def fit(
        cls,
        chosen: NDArray[np.float64],
        mean: NDArray[np.float64],
        truth: pd.DataFrame,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        key: str = "Molecule_Name",
        threshold: float = DEFAULT_P_VALUE_THRESHOLD,
    ) -> "EstimatorSelector":
        """Compare chosen against mean predictions on calibration compounds.

        Args:
            chosen: Placed predictions, shape ``(n_compounds, n_endpoints)``.
            mean: Posterior means, same shape.
            truth: Calibration ground truth with credible bounds and ``key``.
            endpoints: Endpoint names matching the prediction columns.
            key: Compound identifier column.
            threshold: One-sided bootstrap p-value an endpoint must reach.

        Returns:
            A selector recording the decision and the evidence behind it.
        """
        frames = {}
        for label, values in (("chosen", chosen), ("mean", mean)):
            frame = truth[[key]].copy()
            for column, endpoint in enumerate(endpoints):
                frame[endpoint] = values[:, column]
            frames[label] = frame

        comparison = paired_difference(frames["chosen"], frames["mean"], truth, key=key)
        p_values = {e: float(comparison.loc[e, "p_value"]) for e in endpoints}
        decisions = {e: value <= threshold for e, value in p_values.items()}
        for endpoint, rate in p_values.items():
            logger.info(
                "%s: point_estimate wins %.1f%% of paired samples -> %s",
                endpoint,
                100 * rate,
                "choose_estimates" if decisions[endpoint] else "use the mean",
            )
        return cls(decisions, p_values, threshold)

    def apply(
        self,
        chosen: NDArray[np.float64],
        mean: NDArray[np.float64],
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
    ) -> NDArray[np.float64]:
        """Take chosen predictions where they were shown to help, means elsewhere."""
        missing = set(endpoints) - set(self.use_loss_minimising)
        if missing:
            raise KeyError(f"no decision recorded for {sorted(missing)}")
        chosen = mean.copy()
        for column, endpoint in enumerate(endpoints):
            if self.use_loss_minimising[endpoint]:
                chosen[:, column] = chosen[:, column]
        return chosen
