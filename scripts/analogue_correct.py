"""Apply an analogue-based local correction to a global model's test submission.

The graph gives a submission; this nudges each prediction toward the measured training
references the blinded test compounds were built around, shrinking the nudge where there is
no real analogue (see :mod:`cypshift.decision.analogue`). The references are the challenge
training compounds -- their measured values and credible bounds, keyed to the global model's
out-of-fold prediction there, so the ``residual`` estimator corrects a local bias rather than
discarding the global fit. A few estimators and shrinkage settings make a short leaderboard
experiment; none can be judged on the out-of-fold ST-RAE alone.

    python scripts/analogue_correct.py --rev 9102533 --members graph,graph_octant_tox21 \\
        --submission submissions/graph.csv --output submissions/graph_analogue.csv \\
        --estimator residual --shrinkage 1.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.data.challenge import load
from cypshift.decision.analogue import ESTIMATORS, AnalogueCorrector

logger = logging.getLogger("analogue_correct")
BUNDLE_ROOT = Path("data/run_bundles")


def reference_arrays(
    rev: str, members: list[str], seed: int, directory: Path
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SMILES, truth, bounds and the ensemble's out-of-fold prediction over training rows.

    The out-of-fold bundles carry truth, bounds and ``_pred`` per Molecule_Name but no SMILES,
    so the SMILES are joined from the challenge training set.
    """
    frames = []
    for member in members:
        path = BUNDLE_ROOT / rev / f"reg_{member}_seed{seed}" / "out_of_fold.parquet"
        if not path.exists():
            raise SystemExit(f"no out-of-fold bundle for member {member!r} at {path}")
        frames.append(pd.read_parquet(path).set_index("Molecule_Name"))
    reference = frames[0]
    smiles_by_name = load("inhibition", directory).set_index("Molecule_Name")["SMILES"]
    names = reference.index
    smiles = [smiles_by_name.loc[name] for name in names]

    truth = np.column_stack([reference[e].to_numpy(float) for e in REGRESSION_ENDPOINTS])
    lower = np.column_stack(
        [reference[f"{e}_conf_low"].to_numpy(float) for e in REGRESSION_ENDPOINTS]
    )
    upper = np.column_stack(
        [reference[f"{e}_conf_high"].to_numpy(float) for e in REGRESSION_ENDPOINTS]
    )
    ensemble_pred = []
    for endpoint in REGRESSION_ENDPOINTS:
        per_member = np.column_stack(
            [f.loc[names, f"{endpoint}_pred"].to_numpy(float) for f in frames]
        )
        ensemble_pred.append(per_member.mean(axis=1))
    pred = np.column_stack(ensemble_pred)
    return smiles, truth, lower, upper, pred


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", required=True, help="Revision namespace of the member bundles.")
    parser.add_argument("--members", required=True, help="Comma-separated config names (averaged).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--submission", type=Path, required=True, help="Global submission (needs SMILES)."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--estimator", choices=ESTIMATORS, default="residual")
    parser.add_argument("--min-similarity", type=float, default=0.4)
    parser.add_argument("--max-neighbours", type=int, default=5)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--shrinkage", type=float, default=1.0)
    parser.add_argument("--disagreement-scale", type=float, default=1.0)
    arguments = parser.parse_args()

    members = [m.strip() for m in arguments.members.split(",") if m.strip()]
    smiles, truth, lower, upper, pred = reference_arrays(
        arguments.rev, members, arguments.seed, arguments.directory
    )
    corrector = AnalogueCorrector(
        estimator=arguments.estimator,
        min_similarity=arguments.min_similarity,
        max_neighbours=arguments.max_neighbours,
        power=arguments.power,
        shrinkage=arguments.shrinkage,
        disagreement_scale=arguments.disagreement_scale,
    ).fit(smiles, truth, lower, upper, pred)

    submission = pd.read_csv(arguments.submission)
    if "SMILES" not in submission.columns:
        raise SystemExit("the submission must carry a SMILES column to find analogues")
    global_pred = submission[list(REGRESSION_ENDPOINTS)].to_numpy(float)
    corrected = corrector.correct(submission["SMILES"].tolist(), global_pred)

    moved = np.abs(corrected - global_pred)
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        touched = moved[:, column] > 1e-9
        logger.info(
            "%s: %d of %d predictions corrected, mean |shift| %.3f",
            endpoint.split("_")[0],
            int(touched.sum()),
            len(submission),
            float(moved[touched, column].mean()) if touched.any() else 0.0,
        )

    out = submission.copy()
    out[list(REGRESSION_ENDPOINTS)] = corrected
    if not np.isfinite(out[list(REGRESSION_ENDPOINTS)].to_numpy(float)).all():
        raise SystemExit("corrected predictions contain non-finite values")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(arguments.output, index=False)
    logger.info("wrote analogue-corrected submission to %s", arguments.output)


if __name__ == "__main__":
    main()
