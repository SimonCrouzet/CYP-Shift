"""Fit a per-endpoint recalibration on out-of-fold predictions and apply it to a submission.

A model trained on a spread-preserving loss can still under- or over-spread under the
attenuation its regulariser imposes, and the blind test is enriched for potent compounds, so
a per-endpoint map ``y' = a*y + b`` can line the predictions up with the measured band. The
analysis note (notes/performance_analysis_and_next_steps_2026-10-22.md, avenue 1) is pointed
about how to fit it, and the defaults here follow that:

* **Do not assume expansion.** Under squared error the R2-optimal spread ratio is the
  correlation ``rho``, not one, so a good calibration often *shrinks*. The ``affine`` method
  lets the fitted slope move freely below or above one within a restrained band; it does not
  force de-shrinking the way ``variance_match`` does.
* **Do not fit on the high-true-potency stratum.** Selecting rows because their observed
  label is large manufactures positive residuals even for a good conditional-mean predictor,
  so ``fit_potent`` can invent an upward correction that does not transfer. It is kept for
  comparison but is no longer the default; ``affine`` and ``offset`` fit the whole out-of-fold
  set.
* **Calibrate the submitted ensemble, not one member.** ``--members`` averages several
  members' out-of-fold predictions before fitting, so the map is identified on the same
  recipe it is applied to rather than on one member whose scale differs from the blend's.
* **Treat CYP2D6 on its own.** Its labelled training population was selected through CYP2D6
  while its test compounds were selected through other enzymes, so an enrichment-justified
  slope from the other isoforms need not apply. ``--endpoint-method CYP2D6=offset`` gives one
  endpoint a different (gentler) map.

    python scripts/calibrate.py --rev 9102533 --members graph,graph_octant_tox21 \\
        --method affine --endpoint-method CYP2D6=offset \\
        --submission submissions/graph.csv --output submissions/graph_calibrated.csv

None of these can be judged on the out-of-fold ST-RAE, which rewards the compression the
blind test punishes; the calibrated submissions are a leaderboard experiment, so a few are
generated at different methods and compared on the board.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.eval.metrics import soft_threshold_rae

logger = logging.getLogger("calibrate")
BUNDLE_ROOT = Path("data/run_bundles")
METHODS = ("affine", "offset", "fit_potent", "fit_full", "variance_match", "variance_match_rho")


def _fit_soft_threshold(
    truth: np.ndarray,
    pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    free_slope: bool,
    slope_bounds: tuple[float, float],
) -> tuple[float, float]:
    """Slope and intercept minimising ST-RAE, over the intercept alone or over both.

    With ``free_slope`` off the slope is pinned to one and only the offset is fit. With it on,
    the slope is fit too but clamped to ``slope_bounds`` -- a restrained affine, so the map
    cannot run away to a degenerate stretch -- and the offset is re-optimised at the clamped
    slope so the pair stays jointly optimal.
    """
    if not free_slope:
        result = minimize_scalar(lambda b: soft_threshold_rae(truth, pred + b, low, high))
        return 1.0, float(result.x)
    result = minimize(
        lambda p: soft_threshold_rae(truth, p[0] * pred + p[1], low, high),
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
    )
    slope, intercept = float(result.x[0]), float(result.x[1])
    low_bound, high_bound = slope_bounds
    if slope < low_bound or slope > high_bound:
        slope = min(max(slope, low_bound), high_bound)
        re = minimize_scalar(lambda b: soft_threshold_rae(truth, slope * pred + b, low, high))
        intercept = float(re.x)
    return slope, intercept


def affine_for_endpoint(
    truth: np.ndarray,
    pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    method: str,
    potent_quantile: float,
    slope_bounds: tuple[float, float] = (0.25, 2.5),
) -> tuple[float, float]:
    """Slope and intercept for one endpoint, fit by the chosen method."""
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred, low, high = truth[finite], pred[finite], low[finite], high[finite]

    if method == "offset":
        return _fit_soft_threshold(truth, pred, low, high, False, slope_bounds)
    if method == "affine":
        return _fit_soft_threshold(truth, pred, low, high, True, slope_bounds)

    potent = truth >= np.quantile(truth, potent_quantile)
    if method == "variance_match":
        spread = truth.std() / pred.std() if pred.std() > 0 else 1.0
        intercept = truth[potent].mean() - spread * pred[potent].mean()
        return float(spread), float(intercept)
    if method == "variance_match_rho":
        # The R2-optimal spread is rho * sigma_true, NOT the full sigma_true that
        # variance_match targets: de-shrinking all the way overshoots whenever the ranking
        # is imperfect (rho < 1), which it always is.
        usable = pred.std() > 0 and truth.std() > 0
        rho = float(np.corrcoef(truth, pred)[0, 1]) if usable else 0.0
        rho = max(rho, 0.0)
        spread = rho * truth.std() / pred.std() if pred.std() > 0 else 1.0
        intercept = truth[potent].mean() - spread * pred[potent].mean()
        return float(spread), float(intercept)

    # fit_potent / fit_full: free affine minimising ST-RAE on the potent stratum or the whole
    # set. fit_potent is the historic lever the note warns can invent an upward correction.
    rows = np.ones_like(truth, dtype=bool) if method == "fit_full" else potent
    best = minimize(
        lambda p: soft_threshold_rae(truth[rows], p[0] * pred[rows] + p[1], low[rows], high[rows]),
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
    )
    return float(best.x[0]), float(best.x[1])


def load_out_of_fold(rev: str, members: list[str], seed: int) -> pd.DataFrame:
    """One out-of-fold frame, averaging the named members' predictions row for row.

    The truth and bound columns are shared across members (same holdout), so they are taken
    from the first; only the ``_pred`` columns are averaged, which is the prediction the
    submitted ensemble makes.
    """
    frames = []
    for member in members:
        path = BUNDLE_ROOT / rev / f"reg_{member}_seed{seed}" / "out_of_fold.parquet"
        if not path.exists():
            raise SystemExit(f"no out-of-fold bundle for member {member!r} at {path}")
        frames.append(pd.read_parquet(path).set_index("Molecule_Name"))
    reference = frames[0]
    out = reference.copy()
    for endpoint in REGRESSION_ENDPOINTS:
        column = f"{endpoint}_pred"
        stacked = np.column_stack([f.loc[reference.index, column].to_numpy() for f in frames])
        out[column] = stacked.mean(axis=1)
    return out.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", required=True, help="Revision namespace of the member bundles.")
    parser.add_argument(
        "--members",
        required=True,
        help="Comma-separated config names whose out-of-fold predictions are averaged to fit "
        "the map -- the ensemble the submission is, not one member.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--submission", type=Path, required=True, help="Submission to calibrate.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the result.")
    parser.add_argument("--method", choices=METHODS, default="affine")
    parser.add_argument(
        "--endpoint-method",
        action="append",
        default=[],
        metavar="ENZYME=method",
        help="Override the method for one endpoint, e.g. CYP2D6=offset. Repeatable.",
    )
    parser.add_argument("--potent-quantile", type=float, default=0.5)
    parser.add_argument(
        "--slope-bounds",
        default="0.25,2.5",
        help="Low,high clamp on the fitted slope for the affine method (keeps it restrained).",
    )
    arguments = parser.parse_args()

    members = [m.strip() for m in arguments.members.split(",") if m.strip()]
    overrides = {}
    for item in arguments.endpoint_method:
        enzyme, _, method = item.partition("=")
        if method not in METHODS:
            raise SystemExit(f"endpoint-method {item!r}: {method!r} is not one of {METHODS}")
        overrides[enzyme.strip()] = method
    low_bound, high_bound = (float(x) for x in arguments.slope_bounds.split(","))
    slope_bounds = (low_bound, high_bound)

    oof = load_out_of_fold(arguments.rev, members, arguments.seed)
    submission = pd.read_csv(arguments.submission)
    out = submission.copy()

    for endpoint in REGRESSION_ENDPOINTS:
        name = endpoint.split("_")[0]
        method = overrides.get(name, arguments.method)
        truth = oof[endpoint].to_numpy()
        pred = oof[f"{endpoint}_pred"].to_numpy()
        low = oof[f"{endpoint}_conf_low"].to_numpy()
        high = oof[f"{endpoint}_conf_high"].to_numpy()
        finite = np.isfinite(truth) & np.isfinite(pred)
        a, b = affine_for_endpoint(
            truth, pred, low, high, method, arguments.potent_quantile, slope_bounds
        )
        before = soft_threshold_rae(truth[finite], pred[finite], low[finite], high[finite])
        after = soft_threshold_rae(truth[finite], a * pred[finite] + b, low[finite], high[finite])
        logger.info(
            "%s [%s]: a=%.3f b=%+.3f | OOF ST-RAE %.4f->%.4f | pred sd %.2f->%.2f",
            name,
            method,
            a,
            b,
            before,
            after,
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
