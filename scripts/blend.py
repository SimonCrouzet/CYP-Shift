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
from scipy.optimize import minimize, nnls

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.eval.metrics import soft_threshold_rae
from cypshift.eval.scoring import score_submission

logger = logging.getLogger("blend")
BUNDLE_ROOT = Path("data/run_bundles")
_DIR = re.compile(r"^reg_(?P<name>.+)_seed(?P<seed>\d+)$")
#: ``equal`` averages; ``nnls`` least-squares to the point truth; ``st_rae`` fits the weights
#: to the soft-threshold interval error.
METHODS = ("equal", "nnls", "st_rae")


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


def fit_weights(
    stack: np.ndarray,
    truth: np.ndarray,
    method: str,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    """Per-endpoint member weights. ``stack`` is (compounds, members) over labelled rows.

    ``equal`` averages, ``nnls`` is non-negative least squares to the point truth, and
    ``st_rae`` minimises the soft-threshold interval error directly over the simplex of
    non-negative weights summing to one -- the metric the leaderboard scores, which the
    squared-error fit only approximates.
    """
    n_members = stack.shape[1]
    uniform = np.full(n_members, 1.0 / n_members)
    if method == "equal" or n_members == 1:
        return uniform
    if method == "nnls":
        weights, _ = nnls(stack, truth)
        total = weights.sum()
        return weights / total if total > 0 else uniform
    if method == "st_rae":
        if low is None or high is None:
            raise ValueError("st_rae weighting needs the credible bounds")

        def objective(w: np.ndarray) -> float:
            return soft_threshold_rae(truth, stack @ w, low, high)

        result = minimize(
            objective,
            x0=uniform,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_members,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        )
        total = float(result.x.sum())
        return result.x / total if total > 0 else uniform
    raise ValueError(f"unknown weighting method {method!r}")


def per_endpoint_best(members: dict[str, pd.DataFrame]) -> dict[str, str]:
    """For each endpoint, the single member with the best per-endpoint out-of-fold ST-RAE.

    ``--per-endpoint`` uses this to build a per-endpoint specialist blend (one member each).
    """
    best = {}
    for endpoint in REGRESSION_ENDPOINTS:
        scores = {}
        for name, frame in members.items():
            y = frame[endpoint].to_numpy()
            pred = frame[f"{endpoint}_pred"].to_numpy()
            keep = np.isfinite(y) & np.isfinite(pred)
            scores[name] = soft_threshold_rae(
                y[keep],
                pred[keep],
                frame[f"{endpoint}_conf_low"].to_numpy()[keep],
                frame[f"{endpoint}_conf_high"].to_numpy()[keep],
            )
        best[endpoint.split("_")[0]] = min(scores, key=scores.get)
    return best


def blend(
    members: dict[str, pd.DataFrame],
    method: str,
    endpoint_candidates: dict[str, list[str]] | None = None,
    fit_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit per-endpoint weights; return the blended predictions, the truth frame, and weights.

    ``endpoint_candidates`` restricts which members are eligible on a given enzyme (so one
    endpoint can be served by a specialist subset); members not listed get weight zero there.
    ``fit_mask`` (a boolean over the reference row order) restricts the rows the weights are
    *fit* on, so the caller can score on the held-out complement.
    """
    all_names = list(members)
    truth = members[all_names[0]][
        [
            "Molecule_Name",
            *[c for e in REGRESSION_ENDPOINTS for c in (e, f"{e}_conf_low", f"{e}_conf_high")],
        ]
    ].copy()
    predictions = truth[["Molecule_Name"]].copy()
    weights = {}
    for endpoint in REGRESSION_ENDPOINTS:
        enzyme = endpoint.split("_")[0]
        names = (endpoint_candidates or {}).get(enzyme, all_names)
        # Align every candidate member's prediction to the reference row order.
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
        low = truth[f"{endpoint}_conf_low"].to_numpy()
        high = truth[f"{endpoint}_conf_high"].to_numpy()
        labelled = np.isfinite(y)
        fit_rows = labelled if fit_mask is None else (labelled & fit_mask)
        if not fit_rows.any():
            # A sparse endpoint can be emptied by the fit/score split. Fall back to EQUAL
            # weights (label-free) rather than fitting on all labelled rows, which would pull
            # the held-out evaluation rows into the weight fit and defeat the split.
            w = fit_weights(preds[labelled], y[labelled], "equal")
        else:
            w = fit_weights(preds[fit_rows], y[fit_rows], method, low[fit_rows], high[fit_rows])
        # Full-precision weights: rounding here and then applying the rounded dict to the test
        # submissions makes the exported blend differ from the one scored (they need not sum to
        # one). Round only for display.
        weights[enzyme] = {n: float(wi) for n, wi in zip(names, w, strict=False)}
        predictions[endpoint] = preds @ w
    return predictions, truth, weights


def scaffold_fit_mask(
    names: pd.Series, directory: Path, holdout_frac: float, seed: int
) -> np.ndarray:
    """A boolean over ``names`` marking rows used to *fit* the weights, grouped by scaffold.

    Whole Bemis-Murcko scaffolds go to the fit or the score side together, so an analog
    series never straddles the two and the held-out score is not flattered by a near-duplicate
    whose weight was fit in-sample. SMILES come from the challenge training set, keyed by
    Molecule_Name.
    """
    from cypshift.data.challenge import load
    from cypshift.data.splits import murcko_scaffold

    smiles_by_name = load("inhibition", directory).set_index("Molecule_Name")["SMILES"]
    scaffolds = np.array(
        [murcko_scaffold(smiles_by_name.loc[name]) for name in names], dtype=object
    )
    rng = np.random.default_rng(seed)
    unique = list(dict.fromkeys(scaffolds))
    rng.shuffle(unique)
    score = np.zeros(len(names), dtype=bool)
    target = int(round(holdout_frac * len(names)))
    for scaffold in unique:
        if score.sum() >= target:
            break
        score |= scaffolds == scaffold
    logger.info(
        "fit/score split: %d fit, %d score rows (scaffold-grouped)",
        int((~score).sum()),
        int(score.sum()),
    )
    return ~score


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
    parser.add_argument(
        "--per-endpoint",
        action="store_true",
        help="Give each endpoint the single member with the best per-endpoint OOF ST-RAE -- an "
        "endpoint-specialist blend, rather than one macro winner for all four.",
    )
    parser.add_argument(
        "--endpoint-members",
        default=None,
        help="Restrict the candidate members per endpoint, e.g. "
        "'CYP2C9=graph_chemeleon|graph,CYP2D6=graph_abs'. Endpoints left out use every member.",
    )
    # Equal-weight by default: fitting weights (nnls) overfits the small out-of-fold set and
    # the ADMET-challenge write-ups repeatedly found equal averaging beats stacking.
    parser.add_argument("--method", choices=METHODS, default="equal")
    parser.add_argument(
        "--eval-holdout-frac",
        type=float,
        default=0.0,
        help="Fit the weights on a scaffold-grouped fraction of the OOF and SCORE on the held-out "
        "complement, so the blend is assessed off the data its weights were fit on. 0 scores in "
        "sample (the mild-stacking default).",
    )
    parser.add_argument(
        "--directory", type=Path, default=Path("data/raw"), help="Challenge data dir."
    )
    parser.add_argument(
        "--submissions",
        default=None,
        help="Apply the fitted weights to member test submissions: 'name=path,name=path,...'.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Where to write the blend.")
    arguments = parser.parse_args()
    rev = arguments.rev or latest_rev()

    available = load_members(rev, arguments.seed)

    # Establish the fit/score split FIRST, so every label-dependent choice -- member selection
    # as much as weight fitting -- is made on the fitting side only. Selecting on the whole
    # table and then restricting only the weight fit would let the held-out labels help pick
    # the winner, and the "held-out" score would not be independent.
    fit_mask = None
    fit_names: set | None = None
    held_names: set | None = None
    if arguments.eval_holdout_frac > 0:
        reference_names = next(iter(available.values()))["Molecule_Name"]
        fit_mask = scaffold_fit_mask(
            reference_names, arguments.directory, arguments.eval_holdout_frac, arguments.seed
        )
        fit_names = set(reference_names[fit_mask])
        held_names = set(reference_names[~fit_mask])

    def on_rows(members_dict: dict[str, pd.DataFrame], keep: set | None) -> dict[str, pd.DataFrame]:
        if keep is None:
            return members_dict
        return {n: df[df["Molecule_Name"].isin(keep)] for n, df in members_dict.items()}

    selectable = on_rows(available, fit_names)  # fit side only when evaluating held-out
    singles = rank_single(selectable)
    endpoint_candidates: dict[str, list[str]] | None = None
    if arguments.per_endpoint:
        best = per_endpoint_best(selectable)
        endpoint_candidates = {enzyme: [name] for enzyme, name in best.items()}
        chosen = list(dict.fromkeys(best.values()))
    elif arguments.endpoint_members:
        endpoint_candidates = {}
        for pair in arguments.endpoint_members.split(","):
            enzyme, _, spec = pair.partition("=")
            endpoint_candidates[enzyme.strip()] = [m.strip() for m in spec.split("|") if m.strip()]
        chosen = list(dict.fromkeys(m for names in endpoint_candidates.values() for m in names))
        missing = [m for m in chosen if m not in available]
        if missing:
            raise SystemExit(f"no bundle for: {missing}; have {list(available)[:10]}...")
    elif arguments.members:
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
        raise SystemExit(
            "pass --members, --top-k, --per-family, --per-endpoint, or --endpoint-members"
        )
    members = {n: available[n] for n in chosen}

    predictions, truth, weights = blend(members, arguments.method, endpoint_candidates, fit_mask)
    if fit_mask is not None:
        # Score only the held-out rows, which the weights never saw, and compare against the
        # best single member on those same rows -- not its full-table score.
        held = ~fit_mask
        truth = truth.loc[held]
        predictions = predictions.loc[held]
        held_singles = rank_single(on_rows({n: available[n] for n in chosen}, held_names))
        best_single, best_name = held_singles.min(), held_singles.idxmin()
    else:
        best_single, best_name = singles[chosen].min(), singles[chosen].idxmin()
    scores = score_submission(predictions, truth)
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
        print(f"  {iso}: {{{', '.join(f'{n}: {v:.3f}' for n, v in w.items())}}}")

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
