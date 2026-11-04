"""Score a hinge-ridge correction of an incumbent, per enzyme, from the runs' out-of-fold tables.

    python scripts/stack.py --rev questions3-m1 \\
        --incumbent NAME@0+NAME@1+NAME@2 \\
        --member b=B_NAME --member b_tuned=B_TUNED_NAME \\
        --auxiliary B_TUNED_NAME \\
        --output data/stack/b.csv --weights data/stack/b.json

The incumbent is the equal-weight (or ``:WEIGHT``) blend of the cells named; a cell is
``[REV/]NAME[@SEED]``, and ``NAME!`` its pooled members (see ``export_cell_metrics.py``). Each
``--member LABEL=CELL+CELL`` is another blend whose prediction, minus the incumbent's, becomes one
column of the correction. ``--auxiliary CELL`` reads that cell's ``out_of_fold_auxiliary.parquet``
(the other heads of the graph): each potency head enters as its difference from the cell's own
direct head, each response head as it is. Every column is standardised, so a weight is a shift in
pIC50 per standard deviation of the column.

For each enzyme the correction is fitted by ``cypshift.eval.stacking``, with the penalty chosen on
inner folds and the scores taken from outer folds of the same scaffold partition the runs used, so
no compound is corrected by weights that saw its fold. Rows: the incumbent, then each of
``members``, ``auxiliary`` and ``members+auxiliary`` (those that have columns), each raw and with
its predictions rescaled per fold to the incumbent's mean and spread. Pearson and MAE first;
ST-RAE is the interval loss the correction minimises.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_cell_metrics import BUNDLES, ENZYMES, blend, read_cell, split_member  # noqa: E402

from cypshift.config import REGRESSION_ENDPOINTS  # noqa: E402 -- after the src path insert
from cypshift.data.assembly import (  # noqa: E402
    RESPONSE_ASSAY,
    SCREEN_LOG2FC_ASSAY,
    SCREEN_RESPONSE_ASSAY,
    TDI_LABEL_ASSAY,
)
from cypshift.data.challenge import load  # noqa: E402
from cypshift.data.splits import stratified_scaffold_folds  # noqa: E402
from cypshift.eval.metrics import soft_threshold_rae  # noqa: E402
from cypshift.eval.stacking import (  # noqa: E402
    L2_GRID,
    cross_validated_stack,
    fit_stack,
    restore_moments,
)

logger = logging.getLogger("stack")

RESPONSE_ASSAYS = (RESPONSE_ASSAY, SCREEN_RESPONSE_ASSAY, SCREEN_LOG2FC_ASSAY, TDI_LABEL_ASSAY)
DIRECT_ASSAY = "direct_inhibition_drc"
N_FOLDS = 5


def blend_of(rev: str, spec: str, seed: int) -> pd.DataFrame:
    """The weighted blend of ``CELL[:WEIGHT]+CELL[:WEIGHT]...`` out-of-fold tables."""
    parts = [split_member(part, rev, seed) for part in spec.split("+")]
    tables = [read_cell(r, name, s) for r, name, s, _ in parts]
    return blend(tables, [weight for *_, weight in parts])


def read_auxiliary(rev: str, spec: str, seed: int) -> pd.DataFrame:
    """The auxiliary heads of a cell, ``[REV/]NAME[@SEED]``."""
    rev, name, seed, _ = split_member(spec, rev, seed)
    return pd.read_parquet(
        BUNDLES / rev / f"reg_{name}_seed{seed}" / "out_of_fold_auxiliary.parquet"
    )


def scaffold_folds(data_dir: Path, seed: int, n_folds: int = N_FOLDS) -> pd.Series:
    """The fold of each training compound in the partition ``run_cv`` used for this seed."""
    train = load("inhibition", data_dir)
    strata = train[list(REGRESSION_ENDPOINTS)].notna().sum(axis=1).to_numpy()
    folds = stratified_scaffold_folds(train["SMILES"].tolist(), strata, n_folds=n_folds, seed=seed)
    return pd.Series(folds, index=train["Molecule_Name"])


def auxiliary_columns(
    auxiliary: pd.DataFrame, target: str, rows: pd.Index
) -> dict[str, np.ndarray]:
    """The heads of one endpoint as correction columns: potency heads minus the cell's direct
    head, response heads as they are. A head with no value is left out."""
    table = auxiliary.set_index("Molecule_Name").reindex(rows)
    heads = {
        column.split("__", 1)[1]: table[column].to_numpy(float)
        for column in table.columns
        if column.startswith(f"{target}__")
    }
    direct = heads.get(DIRECT_ASSAY)
    out = {}
    for assay, values in heads.items():
        if np.isnan(values).all() or assay == DIRECT_ASSAY:
            continue
        if assay in RESPONSE_ASSAYS or direct is None:
            out[assay] = values
        else:
            out[assay] = values - direct
    return out


def endpoint_metrics(
    truth: np.ndarray, prediction: np.ndarray, low: np.ndarray, high: np.ndarray
) -> dict[str, float]:
    return {
        "Pearson": float(pearsonr(prediction, truth)[0]),
        "MAE": float(np.abs(prediction - truth).mean()),
        "ST-RAE": soft_threshold_rae(truth, prediction, low, high),
    }


def evaluate(
    incumbent: pd.DataFrame,
    members: dict[str, pd.DataFrame],
    auxiliary: pd.DataFrame | None,
    folds: pd.Series,
    grid: tuple[float, ...] = L2_GRID,
) -> tuple[pd.DataFrame, dict]:
    """Per-enzyme and macro metrics of the incumbent and of each cross-validated correction, and
    the weights of each correction fitted on all the compounds."""
    table = incumbent.set_index("Molecule_Name")
    rows, weights = [], {}
    for enzyme in ENZYMES:
        target = f"{enzyme}_pIC50_direct_inhibition"
        needed = [target, f"{target}_conf_low", f"{target}_conf_high", f"{target}_pred"]
        usable = table.index[table[needed].notna().all(axis=1)]
        frame = table.loc[usable]
        truth, base = frame[target].to_numpy(float), frame[f"{target}_pred"].to_numpy(float)
        low, high = (
            frame[f"{target}_conf_low"].to_numpy(float),
            frame[f"{target}_conf_high"].to_numpy(float),
        )
        fold = folds.reindex(usable).to_numpy()
        if np.isnan(fold).any():
            raise ValueError(f"{int(np.isnan(fold).sum())} {enzyme} compounds have no fold")
        fold = fold.astype(int)
        groups = {
            "members": {
                label: other.set_index("Molecule_Name")
                .reindex(usable)[f"{target}_pred"]
                .to_numpy(float)
                - base
                for label, other in members.items()
            },
            "auxiliary": (
                auxiliary_columns(auxiliary, target, usable) if auxiliary is not None else {}
            ),
        }
        groups = {name: columns for name, columns in groups.items() if columns}
        if len(groups) == 2:
            groups["members+auxiliary"] = {**groups["members"], **groups["auxiliary"]}
        rows.append(
            {"enzyme": enzyme, "scenario": "incumbent", "restored": False}
            | endpoint_metrics(truth, base, low, high)
        )
        for scenario, columns in groups.items():
            names = [name for name, values in columns.items() if not np.isnan(values).all()]
            matrix = np.column_stack([columns[name] for name in names])
            corrected, chosen = cross_validated_stack(base, matrix, low, high, fold, grid)
            restored = corrected.copy()
            for each in np.unique(fold):
                held = fold == each
                restored[held] = restore_moments(corrected[held], base[held])
            for flag, prediction in ((False, corrected), (True, restored)):
                rows.append(
                    {"enzyme": enzyme, "scenario": scenario, "restored": flag}
                    | endpoint_metrics(truth, prediction, low, high)
                    | {"l2": float(np.median(chosen))}
                )
            full = fit_stack(base, matrix, low, high, fold, grid)
            weights.setdefault(enzyme, {})[scenario] = {
                "columns": names,
                "weights": full.weights.tolist(),
                "l2": full.l2,
                "centre": full.centre.tolist(),
                "scale": full.scale.tolist(),
            }
    result = pd.DataFrame(rows)
    macro = (
        result.groupby(["scenario", "restored"], sort=False)[["Pearson", "MAE", "ST-RAE"]]
        .mean()
        .reset_index()
        .assign(enzyme="macro")
    )
    return pd.concat([result, macro], ignore_index=True), weights


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the run's results namespace")
    parser.add_argument("--incumbent", required=True, help="CELL[:WEIGHT]+CELL... blend")
    parser.add_argument("--member", nargs="*", default=[], help="LABEL=CELL+CELL")
    parser.add_argument("--auxiliary", default=None, help="the cell whose auxiliary heads to use")
    parser.add_argument("--seed", type=int, default=0, help="the partition's seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    incumbent = blend_of(args.rev, args.incumbent, args.seed)
    members = {}
    for item in args.member:
        label, _, spec = item.partition("=")
        members[label] = blend_of(args.rev, spec, args.seed)
    auxiliary = read_auxiliary(args.rev, args.auxiliary, args.seed) if args.auxiliary else None
    folds = scaffold_folds(args.data_dir, args.seed)
    table, weights = evaluate(incumbent, members, auxiliary, folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if args.weights:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        args.weights.write_text(json.dumps(weights, indent=2) + "\n")
    columns = ["enzyme", "scenario", "restored", "Pearson", "MAE", "ST-RAE", "l2"]
    logger.info("\n%s", table[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
