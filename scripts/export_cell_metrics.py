"""Export Pearson, MAE, R², ST-RAE and Spearman of graph cells from their out-of-fold predictions.

    python scripts/export_cell_metrics.py --rev questions3-m1 --names NAME_A,NAME_B \\
        --output notes/cell_metrics.csv

One row per cell (full recipe name). Each metric is computed per enzyme on the compounds that have
a measurement for it, then averaged over the four enzymes, as the leaderboard does. The columns are
the ones ``plot_learning_curves.py --metrics`` reads.

A name may carry its run's namespace, ``REV/NAME``, and a seed, ``NAME@SEED``, when the cells come
from several runs or seeds; in a ``--blend`` member, ``:WEIGHT`` after that sets its weight (default
1; the weights are normalised). With
``--per-enzyme`` the rows are one per cell and enzyme instead. ``--blend LABEL=NAME+NAME`` adds the
weighted average of the named cells' predictions as one more cell.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.eval.metrics import soft_threshold_rae  # noqa: E402 -- after the src path insert

logger = logging.getLogger("export_cell_metrics")

BUNDLES = Path("data/run_bundles")
ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
COLUMNS = ("Pearson", "MAE", "R2", "ST-RAE", "Spearman")


def enzyme_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    """The five metrics of an out-of-fold table, one row per enzyme."""
    rows = []
    for enzyme in ENZYMES:
        target = f"{enzyme}_pIC50_direct_inhibition"
        pred = oof[f"{target}_pred"].to_numpy()
        keep = oof[target].notna().to_numpy() & ~np.isnan(pred)
        truth, pred = oof.loc[keep, target].to_numpy(), pred[keep]
        rows.append(
            {
                "enzyme": enzyme,
                "Pearson": pearsonr(pred, truth)[0],
                "MAE": mean_absolute_error(truth, pred),
                "R2": r2_score(truth, pred),
                "ST-RAE": soft_threshold_rae(
                    truth,
                    pred,
                    oof.loc[keep, f"{target}_conf_low"].to_numpy(),
                    oof.loc[keep, f"{target}_conf_high"].to_numpy(),
                ),
                "Spearman": spearmanr(pred, truth).statistic,
            }
        )
    return pd.DataFrame(rows)


def bundle_metrics(oof: pd.DataFrame) -> dict[str, float]:
    """The five metrics of an out-of-fold table, each averaged over the enzymes."""
    return enzyme_metrics(oof)[list(COLUMNS)].mean().to_dict()


def blend(tables: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """The weighted average (equal by default) of the predictions of out-of-fold tables over the
    same compounds."""
    first = tables[0].set_index("Molecule_Name")
    weight = np.asarray(weights if weights is not None else np.ones(len(tables)), dtype=float)
    weight = weight / weight.sum()
    out = first.copy()
    for column in (c for c in first.columns if c.endswith("_pred")):
        stacked = [t.set_index("Molecule_Name").reindex(first.index)[column] for t in tables]
        out[column] = np.average(stacked, axis=0, weights=weight)
    return out.reset_index()


def split_member(spec: str, rev: str, seed: int) -> tuple[str, str, int, float]:
    """``(rev, name, seed, weight)`` of ``[REV/]NAME[@SEED][:WEIGHT]``; a name ending in ``!`` is
    kept with its ``!`` (see :func:`read_cell`)."""
    spec, _, weight = spec.partition(":")
    spec, _, own_seed = spec.partition("@")
    if "/" in spec:
        rev, _, spec = spec.rpartition("/")
    return rev, spec, int(own_seed) if own_seed else seed, float(weight) if weight else 1.0


def read_cell(rev: str, name: str, seed: int = 0) -> pd.DataFrame:
    """The out-of-fold table of a cell; ``name`` may be ``[REV/]NAME[@SEED]``. ``NAME!`` reads the
    pooled members' predictions of a cell with a per-target fine-tune (``out_of_fold_pooled``)."""
    rev, name, seed, _ = split_member(name, rev, seed)
    pooled = name.endswith("!")
    path = BUNDLES / rev / f"reg_{name.rstrip('!')}_seed{seed}"
    return pd.read_parquet(
        path / ("out_of_fold_pooled.parquet" if pooled else "out_of_fold.parquet")
    )


def cell_metrics(
    rev: str,
    names: list[str],
    seed: int = 0,
    blends: dict[str, list[str]] | None = None,
    per_enzyme: bool = False,
) -> pd.DataFrame:
    """``name`` plus the five metrics, one row per cell (or per cell and enzyme), from
    ``reg_<name>_seed<seed>`` and from the equal-weight ``blends`` of cells."""
    tables = {name: read_cell(rev, name, seed) for name in names}
    for label, members in (blends or {}).items():
        weights = [split_member(member, rev, seed)[3] for member in members]
        tables[label] = blend([read_cell(rev, member, seed) for member in members], weights)
    rows = []
    for name, table in tables.items():
        if per_enzyme:
            rows += [{"name": name, **row} for row in enzyme_metrics(table).to_dict("records")]
        else:
            rows.append({"name": name, **bundle_metrics(table)})
    columns = ["name", "enzyme", *COLUMNS] if per_enzyme else ["name", *COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the run's results namespace")
    parser.add_argument("--names", default="", help="comma-separated full recipe names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-enzyme", action="store_true", help="one row per cell and enzyme")
    parser.add_argument(
        "--blend",
        nargs="*",
        default=[],
        help="LABEL=NAME+NAME: the equal-weight average of the cells' predictions, as one cell",
    )
    args = parser.parse_args()

    blends = {
        label: members.split("+") for label, members in (item.split("=", 1) for item in args.blend)
    }
    table = cell_metrics(
        args.rev, args.names.split(",") if args.names else [], args.seed, blends, args.per_enzyme
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("wrote %d rows to %s", len(table), args.output)
    logger.info("\n%s", table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
