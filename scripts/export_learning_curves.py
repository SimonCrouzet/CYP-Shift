"""Export the per-epoch learning curves of graph cells to a CSV in ``notes/``.

A cell run with ``CYP_MONITOR=1 CYP_EARLY_STOP=0`` records, for each cross-validation fold, the
mean absolute error of its held-out compounds after every epoch (``curves.json`` in the run bundle).
The training loss per epoch is only in the sweep log, so the log lines are read too. A cell that
is still running has no bundle yet; its held-out errors are then read from the log's ``fold MAE``
column, for the folds that have finished.

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
# A run of several cells at once tags each line with the cell's name; a single-cell run does not.
FOLD = re.compile(r"^\S+ (?:\[([^\]]+)\] )?fold (\d+)/(\d+) \|")
EPOCH = re.compile(
    r"^\S+ (?:\[([^\]]+)\] )?member (\d+)/\d+ stage (\d+)/\d+ epoch (\d+)/(\d+)"
    r"\s+train ([\d.]+)(?:\s+fold MAE ([\d.]+))?"
)


def train_losses(
    log: str, only: set[str] | None = None, untagged: str | None = None
) -> pd.DataFrame:
    """``name, fold, member, stage, epoch, train_loss`` for every epoch line of the log, plus
    ``log_mae`` (the monitor's held-out error, when logged) and ``total`` (the stage's epochs); the
    fold is the one the cell's latest ``fold k/n`` line announced. Lines without a cell tag (a log
    of a single-cell run) belong to ``untagged``."""
    fold: dict[str, int] = {}
    rows = []
    for line in log.splitlines():
        start = FOLD.match(line)
        if start:
            fold[start.group(1) or untagged] = int(start.group(2)) - 1
            continue
        epoch = EPOCH.match(line)
        name = epoch.group(1) or untagged if epoch else None
        if epoch and name in fold and (only is None or name in only):
            rows.append(
                {
                    "name": name,
                    "fold": fold[name],
                    "member": int(epoch.group(2)),
                    "stage": int(epoch.group(3)),
                    "epoch": int(epoch.group(4)),
                    "train_loss": float(epoch.group(6)),
                    "log_mae": float(epoch.group(7)) if epoch.group(7) else None,
                    "total": int(epoch.group(5)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["name", "fold", "member", "stage", "epoch", "train_loss", "log_mae", "total"],
    )


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


def curves(
    rev: str,
    names: list[str],
    logs: str | list[tuple[str, str | None]],
    seed: int = 0,
    keep_partial: bool = False,
) -> pd.DataFrame:
    """The merged curves of the named cells (full recipe names), seed ``seed``. ``logs`` is one log
    text, or pairs of a log text and the cell its untagged lines belong to. A fold a running cell
    has not finished is left out unless ``keep_partial``."""
    if isinstance(logs, str):
        logs = [(logs, None)]
    keys = ["name", "fold", "member", "stage", "epoch"]
    errors = pd.concat(
        [fold_errors(BUNDLES / rev / f"reg_{name}_seed{seed}", name) for name in names],
        ignore_index=True,
    )
    losses = pd.concat(
        [train_losses(text, set(names), untagged) for text, untagged in logs], ignore_index=True
    )
    table = errors.merge(losses, on=keys, how="outer").sort_values(keys).reset_index(drop=True)
    # A cell still running has no bundle: take the log's errors, and only for finished folds.
    table["fold_mae"] = table["fold_mae"].fillna(table["log_mae"])
    last = table.groupby(["name", "fold", "member", "stage"])["epoch"].transform("max")
    unfinished = table["total"].notna() & (last < table["total"]) & (not keep_partial)
    return table[~unfinished].drop(columns=["log_mae", "total"]).reset_index(drop=True)


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
    parser.add_argument(
        "--log",
        nargs="*",
        default=[],
        help="sweep log files; PATH=NAME gives the cell an untagged single-cell log belongs to",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--partial", action="store_true", help="keep the epochs so far of a fold still training"
    )
    args = parser.parse_args()

    logs = []
    for item in args.log:
        path, _, cell = item.partition("=")
        logs.append((Path(path).read_text(errors="replace"), cell or None))
    table = curves(args.rev, args.names.split(","), logs, args.seed, args.partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    logger.info("wrote %d rows to %s", len(table), args.output)
    logger.info("\n%s", summary(table).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
