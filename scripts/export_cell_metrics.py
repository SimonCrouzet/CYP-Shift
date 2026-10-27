"""Export Pearson, MAE, R², ST-RAE and Spearman of graph cells from their out-of-fold predictions.

    python scripts/export_cell_metrics.py --rev questions3-m1 --names NAME_A,NAME_B \\
        --output notes/cell_metrics.csv

One row per cell (full recipe name). Each metric is computed per enzyme on the compounds that have
a measurement for it, then averaged over the four enzymes, as the leaderboard does. The columns are
the ones ``plot_learning_curves.py --metrics`` reads.
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


def bundle_metrics(oof: pd.DataFrame) -> dict[str, float]:
    """The five metrics of an out-of-fold table, each averaged over the enzymes."""
    rows = []
    for enzyme in ENZYMES:
        target = f"{enzyme}_pIC50_direct_inhibition"
        pred = oof[f"{target}_pred"].to_numpy()
        keep = oof[target].notna().to_numpy() & ~np.isnan(pred)
        truth, pred = oof.loc[keep, target].to_numpy(), pred[keep]
        rows.append(
            (
                pearsonr(pred, truth)[0],
                mean_absolute_error(truth, pred),
                r2_score(truth, pred),
                soft_threshold_rae(
                    truth,
                    pred,
                    oof.loc[keep, f"{target}_conf_low"].to_numpy(),
                    oof.loc[keep, f"{target}_conf_high"].to_numpy(),
                ),
                spearmanr(pred, truth).statistic,
            )
        )
    return dict(zip(COLUMNS, np.mean(rows, axis=0).tolist(), strict=True))


def cell_metrics(rev: str, names: list[str], seed: int = 0) -> pd.DataFrame:
    """``name`` plus the five metrics, one row per cell, from ``reg_<name>_seed<seed>``."""
    rows = []
    for name in names:
        path = BUNDLES / rev / f"reg_{name}_seed{seed}" / "out_of_fold.parquet"
        rows.append({"name": name, **bundle_metrics(pd.read_parquet(path))})
    return pd.DataFrame(rows, columns=["name", *COLUMNS])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the run's results namespace")
    parser.add_argument("--names", required=True, help="comma-separated full recipe names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = cell_metrics(args.rev, args.names.split(","), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("wrote %d rows to %s", len(table), args.output)
    logger.info("\n%s", table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
