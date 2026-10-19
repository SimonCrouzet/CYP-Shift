"""Blend a sweep's models by fitting per-endpoint weights on their out-of-fold predictions.

A single model is capped by its own inductive bias; combining models with different biases
(a boosted tree, a graph D-MPNN, a curve) usually ranks better and, when the members
disagree, spreads the blend out -- which fights the prediction compression that costs us on
the leaderboard. The weights are fit on the cross-validation out-of-fold predictions the
sweep already wrote (``data/run_bundles/<rev>/reg_<name>_seed<seed>/out_of_fold.parquet``),
so no member ever sees its own blended target in-sample at the base level.

Everything is a parameter: which members, how many, the fitting method, and the revision.

    # the three named members, non-negative least squares per endpoint
    python scripts/blend.py --members graph,boosted_octant_tox21_deep,curve_heads --method nnls
    # the best 5 single models by macro ST-RAE, equally weighted
    python scripts/blend.py --top-k 5 --method equal

Weights fit on out-of-fold predictions are mild stacking: the meta-weights do see all folds,
so the reported blend score is a little optimistic. It is still the right relative read of
whether a blend beats its best member; treat the absolute level with the usual caution.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.eval.scoring import score_submission

logger = logging.getLogger("blend")
BUNDLE_ROOT = Path("data/run_bundles")
_DIR = re.compile(r"^reg_(?P<name>.+)_seed(?P<seed>\d+)$")
METHODS = ("equal", "nnls")


def latest_rev() -> str:
    revs = [p for p in BUNDLE_ROOT.iterdir() if p.is_dir()]
    if not revs:
        raise SystemExit(f"no run bundles under {BUNDLE_ROOT}")
    return max(revs, key=lambda p: p.stat().st_mtime).name


def load_members(rev: str, seed: int) -> dict[str, pd.DataFrame]:
    """Each regression config's out-of-fold frame for this seed, keyed by config name."""
    root = BUNDLE_ROOT / rev
    members = {}
    for bundle in sorted(root.glob(f"reg_*_seed{seed}")):
        match = _DIR.match(bundle.name)
        oof = bundle / "out_of_fold.parquet"
        if match and oof.exists():
            members[match["name"]] = pd.read_parquet(oof)
    if not members:
        raise SystemExit(f"no regression bundles for seed {seed} under {root}")
    return members


def _truth_columns() -> list[str]:
    """Molecule_Name plus each endpoint's value and credible-interval bounds (no predictions)."""
    cols = ["Molecule_Name"]
    for endpoint in REGRESSION_ENDPOINTS:
        cols += [endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]
    return cols


def rank_single(members: dict[str, pd.DataFrame]) -> pd.Series:
    """Each member's own macro ST-RAE, best first -- the baseline a blend must beat."""
    scores = {}
    for name, frame in members.items():
        predictions = frame[["Molecule_Name", *[f"{e}_pred" for e in REGRESSION_ENDPOINTS]]].rename(
            columns={f"{e}_pred": e for e in REGRESSION_ENDPOINTS}
        )
        scores[name] = score_submission(predictions, frame[_truth_columns()]).loc[
            MACRO_LABEL, "ST-RAE_mean"
        ]
    return pd.Series(scores).sort_values()


def fit_weights(stack: np.ndarray, truth: np.ndarray, method: str) -> np.ndarray:
    """Per-endpoint member weights. ``stack`` is (compounds, members) over labelled rows."""
    if method == "equal":
        return np.full(stack.shape[1], 1.0 / stack.shape[1])
    weights, _ = nnls(stack, truth)  # non-negative least squares to the truth
    total = weights.sum()
    return weights / total if total > 0 else np.full(stack.shape[1], 1.0 / stack.shape[1])


def blend(members: dict[str, pd.DataFrame], method: str) -> tuple[pd.DataFrame, dict]:
    """Fit per-endpoint weights; return the blended predictions, the truth frame, and weights."""
    names = list(members)
    truth = members[names[0]][
        [
            "Molecule_Name",
            *[c for e in REGRESSION_ENDPOINTS for c in (e, f"{e}_conf_low", f"{e}_conf_high")],
        ]
    ].copy()
    predictions = truth[["Molecule_Name"]].copy()
    weights = {}
    for endpoint in REGRESSION_ENDPOINTS:
        # Align every member's prediction to the reference row order.
        preds = np.column_stack(
            [
                members[n]
                .set_index("Molecule_Name")
                .loc[truth["Molecule_Name"], f"{endpoint}_pred"]
                .to_numpy()
                for n in names
            ]
        )
        y = truth[endpoint].to_numpy()
        labelled = np.isfinite(y)
        w = fit_weights(preds[labelled], y[labelled], method)
        weights[endpoint.split("_")[0]] = {
            n: round(float(wi), 3) for n, wi in zip(names, w, strict=False)
        }
        predictions[endpoint] = preds @ w
    return predictions, truth, weights


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=None, help="Revision namespace; default is the latest.")
    parser.add_argument("--seed", type=int, default=0, help="Which CV repeat's bundles to use.")
    parser.add_argument("--members", default=None, help="Comma-separated config names to blend.")
    parser.add_argument("--top-k", type=int, default=None, help="Blend the best K single models.")
    parser.add_argument(
        "--per-family",
        action="store_true",
        help="Blend the single best model from each family (graph/boosted/tfm/neural/curve) -- "
        "architecturally diverse members, which the challenge write-ups found beat more of a kind.",
    )
    # Equal-weight by default: fitting weights (nnls) overfits the small out-of-fold set and
    # the ADMET-challenge write-ups repeatedly found equal averaging beats stacking.
    parser.add_argument("--method", choices=METHODS, default="equal")
    parser.add_argument(
        "--submissions",
        default=None,
        help="Apply the fitted weights to member test submissions: 'name=path,name=path,...'.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Where to write the blend.")
    arguments = parser.parse_args()
    rev = arguments.rev or latest_rev()

    available = load_members(rev, arguments.seed)
    singles = rank_single(available)
    if arguments.members:
        chosen = [m.strip() for m in arguments.members.split(",")]
        missing = [m for m in chosen if m not in available]
        if missing:
            raise SystemExit(f"no bundle for: {missing}; have {list(available)[:10]}...")
    elif arguments.per_family:
        families = ("ref", "graph", "neural", "curve", "tfm", "boosted", "per_target")

        def family(name: str) -> str:
            return next((f for f in families if name.startswith(f)), "other")

        best_of: dict[str, str] = {}
        for name in singles.index:  # singles is sorted best-first, so the first per family wins
            best_of.setdefault(family(name), name)
        chosen = list(best_of.values())
    elif arguments.top_k:
        chosen = list(singles.head(arguments.top_k).index)
    else:
        raise SystemExit("pass --members, --top-k, or --per-family")
    members = {n: available[n] for n in chosen}

    predictions, truth, weights = blend(members, arguments.method)
    scores = score_submission(predictions, truth)
    best_single = singles[chosen].min()
    best_name = singles[chosen].idxmin()
    print(f"=== rev {rev}: blend of {len(members)} members ({arguments.method}) ===")
    print(f"members: {chosen}")
    print(
        f"blend  macro ST-RAE {scores.loc[MACRO_LABEL, 'ST-RAE_mean']:.4f} | "
        f"macro Spearman {scores.loc[MACRO_LABEL, 'Spearman_R_mean']:.4f}"
    )
    print(f"best single ({best_name}) macro ST-RAE {best_single:.4f}")
    print(
        "per-endpoint ST-RAE:",
        {e.split("_")[0]: round(scores.loc[e, "ST-RAE_mean"], 4) for e in REGRESSION_ENDPOINTS},
    )
    print("weights per endpoint:")
    for iso, w in weights.items():
        print(f"  {iso}: {w}")

    if arguments.submissions and arguments.output:
        apply_to_submissions(arguments.submissions, weights, arguments.output)


def apply_to_submissions(spec: str, weights: dict, output: Path) -> None:
    """Combine member test submissions with the fitted per-endpoint weights, then write.

    ``spec`` is 'name=path,name=path,...'; every member that carries weight for an endpoint
    must have a submission, and all submissions must cover the same compounds.
    """
    paths = dict(pair.split("=", 1) for pair in spec.split(","))
    subs = {name: pd.read_csv(path).set_index("Molecule_Name") for name, path in paths.items()}
    reference = next(iter(subs.values()))
    out = reference.reset_index()[["Molecule_Name"]].copy()
    if "SMILES" in reference.columns:
        out.insert(0, "SMILES", reference["SMILES"].to_numpy())
    for endpoint in REGRESSION_ENDPOINTS:
        iso = endpoint.split("_")[0]
        blended = np.zeros(len(out))
        for name, w in weights[iso].items():
            if w == 0:
                continue
            if name not in subs:
                raise SystemExit(f"member {name!r} has weight on {iso} but no submission given")
            blended += w * subs[name].loc[out["Molecule_Name"], endpoint].to_numpy()
        out[endpoint] = blended
    values = out[list(REGRESSION_ENDPOINTS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise SystemExit("blended submission contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"wrote blended submission ({len(out)} rows) to {output}")


if __name__ == "__main__":
    main()
