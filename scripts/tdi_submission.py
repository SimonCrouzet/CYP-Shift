"""Write a TDI-track submission from the label heads of graph cells run on the TDI folds.

    python scripts/tdi_submission.py --rev questions3-m1 \\
        --names focus_tdi5_label_b_reg_uncovered_0_8352aa --output submissions/tdi.csv

A cell's five fold models each predict the blinded test set (``test_by_fold_auxiliary.parquet``);
the score of a compound is the mean of the ``is_TDI`` label head over the folds. With several
``--names`` the cells' scores are averaged with equal weight.

The call needs a threshold. ``oof`` (default) takes the score at which the cell's out-of-fold
scores of the compounds with a direct measurement flag their training prevalence as positive, and
applies that score to the test set, so the number of test calls follows the test scores. ``share``
flags the same prevalence share of the test set instead. The report gives the out-of-fold MCC and
AUC of the scores and the number of calls per enzyme; a JSON next to the CSV records the recipe.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.data.challenge import load  # noqa: E402 -- after the src path insert

logger = logging.getLogger("tdi_submission")

BUNDLES = Path("data/run_bundles")
ENZYMES = ("CYP2D6", "CYP3A4")


def label_column(enzyme: str) -> str:
    return f"{enzyme}_pIC50_direct_inhibition__tdi_label"


def mean_over_folds(table: pd.DataFrame, column: str) -> pd.Series:
    """The mean of a column over the fold models, per compound."""
    return table.groupby("Molecule_Name")[column].mean()


def calls(
    oof: np.ndarray, truth: np.ndarray, test: np.ndarray, rule: str = "oof"
) -> tuple[np.ndarray, float]:
    """Boolean test calls, and the score threshold they use.

    ``oof`` and ``truth`` are the out-of-fold scores and 0/1 labels of the scored compounds. The
    prevalence of ``truth`` is the share flagged: ``oof`` applies the score that flags that share
    of ``oof`` to ``test``; ``share`` flags that share of ``test``.
    """
    prevalence = float(np.mean(truth))
    if rule == "oof":
        threshold = float(np.quantile(oof, 1.0 - prevalence))
    elif rule == "share":
        threshold = float(np.quantile(test, 1.0 - prevalence))
    else:
        raise ValueError(f"rule is 'oof' or 'share', not {rule!r}")
    return test >= threshold, threshold


def out_of_fold_report(oof: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """MCC at the prevalence share (the call the submission makes) and AUC of out-of-fold scores."""
    called, _ = calls(oof, truth, oof, "share")
    return {
        "MCC": float(matthews_corrcoef(truth.astype(int), called.astype(int))),
        "AUC": float(roc_auc_score(truth, oof)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the run's results namespace")
    parser.add_argument("--names", required=True, help="comma-separated full recipe names")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rule", choices=("oof", "share"), default="oof")
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = args.names.split(",")
    test = load("test", args.directory)
    labels = load("tdi", args.directory).set_index("Molecule_Name")
    direct = set(load("inhibition", args.directory)["Molecule_Name"])
    submission = test[["SMILES", "Molecule_Name"]].copy()
    summary = {"cells": names, "rule": args.rule, "enzymes": {}}
    for enzyme in ENZYMES:
        column = label_column(enzyme)
        oof_scores, test_scores = [], []
        for name in names:
            path = BUNDLES / args.rev / f"reg_{name}_seed{args.seed}"
            oof_table = pd.read_parquet(path / "out_of_fold_auxiliary.parquet")
            test_table = pd.read_parquet(path / "test_by_fold_auxiliary.parquet")
            oof_scores.append(oof_table.set_index("Molecule_Name")[column])
            test_scores.append(mean_over_folds(test_table, column))
        oof_mean = pd.concat(oof_scores, axis=1).mean(axis=1)
        test_mean = pd.concat(test_scores, axis=1).mean(axis=1).reindex(test["Molecule_Name"])
        truth = labels[f"{enzyme}_is_TDI"].reindex(oof_mean.index)
        scored = truth.notna() & oof_mean.index.isin(direct) & oof_mean.notna()
        y = truth[scored].to_numpy().astype(int)
        called, threshold = calls(oof_mean[scored].to_numpy(), y, test_mean.to_numpy(), args.rule)
        submission[f"{enzyme}_is_TDI"] = called
        report = out_of_fold_report(oof_mean[scored].to_numpy(), y)
        summary["enzymes"][enzyme] = {
            **report,
            "scored_compounds": int(scored.sum()),
            "prevalence": float(y.mean()),
            "threshold": threshold,
            "test_calls": int(called.sum()),
            "test_share": float(called.mean()),
            "test_score_median": float(test_mean.median()),
            "oof_score_median": float(oof_mean[scored].median()),
        }
        logger.info("%s: %s", enzyme, json.dumps(summary["enzymes"][enzyme]))
    assert submission[["CYP2D6_is_TDI", "CYP3A4_is_TDI"]].notna().all().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %d rows to %s", len(submission), args.output)


if __name__ == "__main__":
    main()
