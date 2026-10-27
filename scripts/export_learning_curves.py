"""Export the per-epoch learning curves of graph cells to a CSV in ``notes/``.

A cell run with ``CYP_MONITOR=1 CYP_EARLY_STOP=0`` records, for each cross-validation fold, the
mean absolute error of its held-out compounds after every epoch (``curves.json`` in the run bundle).
The training loss per epoch is only in the sweep log, so the log lines are read too.

    python scripts/export_learning_curves.py --rev questions2-m1 --names NAME_A,NAME_B \\
        --log scratch/sweep_g.log --output notes/learning_curves_capacity.csv

One row per cell, fold and epoch: ``train_loss`` (from the log) and ``fold_mae`` (from the bundle).
The summary printed at the end gives, per cell and averaged over folds, the lowest held-out error,
the epoch it occurs at, the last epoch's error and the last training loss.
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("export_learning_curves")

BUNDLES = Path("data/run_bundles")
FOLD = re.compile(r"^\S+ \[([^\]]+)\] fold (\d+)/(\d+) \|")
EPOCH = re.compile(
    r"^\S+ \[([^\]]+)\] member (\d+)/\d+ stage (\d+)/\d+ epoch (\d+)/\d+\s+train ([\d.]+)"
)


def train_losses(log: str, only: set[str] | None = None) -> pd.DataFrame:
    """``name, fold, member, stage, epoch, train_loss`` for every epoch line of the log; the fold
    is the one the cell's latest ``fold k/n`` line announced."""
    fold: dict[str, int] = {}
    rows = []
    for line in log.splitlines():
        start = FOLD.match(line)
        if start:
            fold[start.group(1)] = int(start.group(2)) - 1
            continue
        epoch = EPOCH.match(line)
        if epoch and epoch.group(1) in fold and (only is None or epoch.group(1) in only):
            rows.append(
                {
                    "name": epoch.group(1),
                    "fold": fold[epoch.group(1)],
                    "member": int(epoch.group(2)),
                    "stage": int(epoch.group(3)),
                    "epoch": int(epoch.group(4)),
                    "train_loss": float(epoch.group(5)),
                }
            )
    return pd.DataFrame(rows, columns=["name", "fold", "member", "stage", "epoch", "train_loss"])


def fold_errors(bundle: Path, name: str) -> pd.DataFrame:
    """``name, fold, member, stage, epoch, fold_mae`` from a bundle's ``curves.json``."""
    path = bundle / "curves.json"
    if not path.exists():
        return pd.DataFrame(columns=["name", "fold", "member", "stage", "epoch", "fold_mae"])
    rows = []
    for fold, entries in json.loads(path.read_text()).items():
        for entry in entries:
            for epoch, error in enumerate(entry["error"], start=1):
                rows.append(
                    {
                        "name": name,
                        "fold": int(fold),
                        "member": entry["member"],
                        "stage": entry["stage"],
                        "epoch": epoch,
                        "fold_mae": error,
                    }
                )
    return pd.DataFrame(rows)


def curves(rev: str, names: list[str], log: str, seed: int = 0) -> pd.DataFrame:
    """The merged curves of the named cells (full recipe names), seed ``seed``."""
    keys = ["name", "fold", "member", "stage", "epoch"]
    errors = pd.concat(
        [fold_errors(BUNDLES / rev / f"reg_{name}_seed{seed}", name) for name in names],
        ignore_index=True,
    )
    losses = train_losses(log, set(names))
    return errors.merge(losses, on=keys, how="outer").sort_values(keys).reset_index(drop=True)


def _epoch_of_lowest(fold: pd.DataFrame) -> int:
    return int(fold.loc[fold["fold_mae"].idxmin(), "epoch"])


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per cell, averaged over folds: lowest held-out error and its epoch, last held-out error and
    last training loss."""
    rows = []
    for name, cell in table.groupby("name"):
        errors = cell.dropna(subset=["fold_mae"]).groupby("fold")
        losses = cell.dropna(subset=["train_loss"]).groupby("fold")["train_loss"]
        rows.append(
            {
                "cell": name,
                "lowest_fold_mae": errors["fold_mae"].min().mean(),
                "epoch_of_lowest": errors.apply(_epoch_of_lowest).mean(),
                "last_fold_mae": errors["fold_mae"].last().mean(),
                "last_train_loss": losses.last().mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rev", required=True, help="the run's results namespace, e.g. questions2-m1"
    )
    parser.add_argument("--names", required=True, help="comma-separated full recipe names")
    parser.add_argument("--log", type=Path, nargs="*", default=[], help="sweep log files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = "\n".join(path.read_text(errors="replace") for path in args.log)
    table = curves(args.rev, args.names.split(","), log, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("wrote %d rows to %s", len(table), args.output)
    logger.info("\n%s", summary(table).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
