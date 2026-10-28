"""Write the TDI cross-validation folds to a table and report them.

    python scripts/tdi_folds.py --output data/tdi_runs/folds_tdi5_seed0.parquet

The folds are ``tdi_scaffold_folds`` of the TDI training table: scaffold-grouped, stratified on the
CYP2D6 and CYP3A4 TDI positives. Models look the folds up by compound (``tdi_baseline.py`` and a
graph run with ``data.tdi_folds``), so this table is a record, not an input.

Per compound the table also carries ``nn_similarity``, its highest Tanimoto similarity to the
compounds outside its fold, and ``in_test_range``, whether that lies within the 5th to 95th
percentile of the blinded test set's own nearest-neighbour similarity to the training compounds. The
report gives per fold the counts of compounds, those with a direct measurement, labelled and
positive per enzyme, and those in the test range.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.data.challenge import load  # noqa: E402 -- after the src path insert
from cypshift.data.splits import (  # noqa: E402
    max_similarity_against_retained,
    max_similarity_to,
    morgan_fingerprints,
    tdi_scaffold_folds,
)

logger = logging.getLogger("tdi_folds")

ENZYMES = ("CYP2D6", "CYP3A4")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tdi = load("tdi", args.directory)
    direct = set(load("inhibition", args.directory)["Molecule_Name"])
    test = load("test", args.directory)
    folds = tdi_scaffold_folds(tdi, args.folds, args.seed)

    fingerprints = morgan_fingerprints(tdi["SMILES"].tolist())
    out = tdi[["Molecule_Name", "SMILES"]].copy()
    out["fold"] = folds.to_numpy()
    out["direct"] = out["Molecule_Name"].isin(direct).to_numpy()
    out["nn_similarity"] = np.nan
    for fold in range(args.folds):
        held = out["fold"].to_numpy() == fold
        out.loc[held, "nn_similarity"] = max_similarity_against_retained(fingerprints, ~held)[held]
    target = max_similarity_to(morgan_fingerprints(test["SMILES"].tolist()), fingerprints)
    low, high = np.quantile(target, [0.05, 0.95])
    out["in_test_range"] = out["nn_similarity"].between(low, high)
    for enzyme in ENZYMES:
        out[f"{enzyme}_is_TDI"] = tdi[f"{enzyme}_is_TDI"].to_numpy()

    rows = []
    for fold, group in out.groupby("fold"):
        row = {"fold": fold, "n": len(group), "direct": int(group["direct"].sum())}
        for enzyme in ENZYMES:
            label = group[f"{enzyme}_is_TDI"]
            covered = group["direct"] & label.notna()
            row[f"{enzyme} labelled"] = int(covered.sum())
            row[f"{enzyme} positive"] = int((covered & (label == 1.0)).sum())
            row[f"{enzyme} positive, test range"] = int(
                (covered & (label == 1.0) & group["in_test_range"]).sum()
            )
        row["in test range"] = int(group["in_test_range"].sum())
        row["median similarity"] = round(float(group["nn_similarity"].median()), 3)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info(
        "test nearest-neighbour similarity: median %.3f, range %.3f-%.3f",
        np.median(target),
        low,
        high,
    )
    logger.info("\n%s", pd.DataFrame(rows).to_string(index=False))
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
