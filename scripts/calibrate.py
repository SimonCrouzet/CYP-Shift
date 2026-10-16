"""Fit a per-endpoint affine recalibration and apply it to a submission.

A model trained on a spread-preserving loss still under-spreads under the attenuation the
regularizer imposes, and the blind test is enriched for potent compounds, so predictions
that sit safely near the training mean land outside the narrow bands of the real actives.
A two-parameter map per endpoint, ``y' = a*y + b``, fixes both: slope ``a>1`` restores the
spread, intercept ``b`` shifts toward the potent end the soft-threshold metric rewards.

The map is fit on the model's out-of-fold predictions (its CV bundle), and -- this is the
point -- it is fit on the **potent stratum** (the compounds that resemble the shifted blind
test), not the whole holdout. Fitting on the full holdout optimises the compression the
metric rewards in-distribution (a dead end we have already measured); fitting where the
truth is extreme forces the map to spread up to reach it.

    python scripts/calibrate.py --rev 87b0cca --member graph \\
        --submission submissions/submission_2026-10-15_graph.csv \\
        --output submissions/submission_2026-10-15_graph_calibrated.csv

Everything is a parameter: the member whose OOF fits the map, the potent quantile, and the
method (``fit_potent`` optimises ST-RAE on the potent stratum; ``variance_match`` sets the
slope from the label/prediction spread ratio and the intercept from the potent-stratum mean;
``variance_match_rho`` scales that slope by the OOF correlation so the spread lands on the
R2-optimal rho*sigma_true rather than the full sigma_true that variance_match overshoots).

None of these can be judged on the out-of-fold ST-RAE, which rewards the compression the
blind test punishes; the calibrated submissions are a leaderboard experiment, so a few are
generated at different methods/quantiles and compared on the board.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.eval.metrics import soft_threshold_rae

logger = logging.getLogger("calibrate")
BUNDLE_ROOT = Path("data/run_bundles")
METHODS = ("fit_potent", "variance_match", "variance_match_rho", "fit_full")


def affine_for_endpoint(
    truth: np.ndarray,
    pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    method: str,
    potent_quantile: float,
) -> tuple[float, float]:
    """Slope and intercept for one endpoint, fit by the chosen method."""
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred, low, high = truth[finite], pred[finite], low[finite], high[finite]
    potent = truth >= np.quantile(truth, potent_quantile)

    if method == "variance_match":
        spread = truth.std() / pred.std() if pred.std() > 0 else 1.0
        intercept = truth[potent].mean() - spread * pred[potent].mean()
        return float(spread), float(intercept)

    if method == "variance_match_rho":
        # The R2-optimal spread is rho * sigma_true, NOT the full sigma_true that
        # variance_match targets: de-shrinking all the way overshoots whenever the ranking
        # is imperfect (rho < 1), which it always is. Scale the spread ratio by the OOF
        # correlation, then place the potent stratum so the active end -- where the
        # soft-threshold metric bites -- lands on its truth, biasing up.
        usable = pred.std() > 0 and truth.std() > 0
        rho = float(np.corrcoef(truth, pred)[0, 1]) if usable else 0.0
        rho = max(rho, 0.0)
        spread = rho * truth.std() / pred.std() if pred.std() > 0 else 1.0
        intercept = truth[potent].mean() - spread * pred[potent].mean()
        return float(spread), float(intercept)

    rows = np.ones_like(truth, dtype=bool) if method == "fit_full" else potent

    def objective(params: np.ndarray) -> float:
        a, b = params
        return soft_threshold_rae(truth[rows], a * pred[rows] + b, low[rows], high[rows])

    best = minimize(objective, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return float(best.x[0]), float(best.x[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", required=True, help="Revision namespace of the member's bundle.")
    parser.add_argument("--member", required=True, help="Config name whose OOF fits the map.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submission", type=Path, required=True, help="Submission to calibrate.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the result.")
    parser.add_argument("--method", choices=METHODS, default="fit_potent")
    parser.add_argument("--potent-quantile", type=float, default=0.5)
    arguments = parser.parse_args()

    oof = pd.read_parquet(
        BUNDLE_ROOT
        / arguments.rev
        / f"reg_{arguments.member}_seed{arguments.seed}"
        / "out_of_fold.parquet"
    )
    submission = pd.read_csv(arguments.submission)
    out = submission.copy()

    for endpoint in REGRESSION_ENDPOINTS:
        truth = oof[endpoint].to_numpy()
        pred = oof[f"{endpoint}_pred"].to_numpy()
        low = oof[f"{endpoint}_conf_low"].to_numpy()
        high = oof[f"{endpoint}_conf_high"].to_numpy()
        finite = np.isfinite(truth) & np.isfinite(pred)
        a, b = affine_for_endpoint(
            truth, pred, low, high, arguments.method, arguments.potent_quantile
        )

        before_full = soft_threshold_rae(truth[finite], pred[finite], low[finite], high[finite])
        after_full = soft_threshold_rae(
            truth[finite], a * pred[finite] + b, low[finite], high[finite]
        )
        potent = finite & (truth >= np.quantile(truth[finite], arguments.potent_quantile))
        before_potent = soft_threshold_rae(truth[potent], pred[potent], low[potent], high[potent])
        after_potent = soft_threshold_rae(
            truth[potent], a * pred[potent] + b, low[potent], high[potent]
        )
        name = endpoint.split("_")[0]
        logger.info(
            "%s: a=%.3f b=%+.3f | ST-RAE full %.4f->%.4f  potent %.4f->%.4f | pred sd %.2f->%.2f",
            name,
            a,
            b,
            before_full,
            after_full,
            before_potent,
            after_potent,
            pred[finite].std(),
            (a * pred[finite] + b).std(),
        )
        out[endpoint] = a * submission[endpoint].to_numpy() + b

    values = out[list(REGRESSION_ENDPOINTS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise SystemExit("calibrated predictions contain non-finite values")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(arguments.output, index=False)
    logger.info("wrote calibrated submission to %s", arguments.output)


if __name__ == "__main__":
    main()
