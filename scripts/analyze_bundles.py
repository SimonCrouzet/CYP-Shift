"""Rank a sweep's configurations by the challenge metric, macro soft-threshold ST-RAE.

ST-RAE is what the leaderboard scores, so it is the objective here too; the comparison is
paired across seeds (every configuration on the same holdouts), and the relative gaps are
what matter since the absolute holdout level runs optimistic (0.66 holdout vs 0.86 live for
the same file). Spearman and R2 are reported beside it as transfer diagnostics -- the
holdout *ranking* does track the leaderboard (0.636 vs 0.669), so a config that wins on
ST-RAE while its Spearman drops deserves a second look. Reads the out-of-fold bundles a
sweep writes (``data/run_bundles/<rev>/<phase>_<name>_seed<seed>/``).

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/analyze_bundles.py            # latest rev
    /mnt/wsl/ext/envs/cypshift/bin/python scripts/analyze_bundles.py --rev 2ef8ec3
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.eval.scoring import score_submission

logger = logging.getLogger("analyze")
BUNDLE_ROOT = Path("data/run_bundles")
_DIR = re.compile(r"^(?P<phase>[^_]+)_(?P<name>.+)_seed(?P<seed>\d+)$")


def latest_rev() -> str:
    """The most recently modified revision namespace under the bundle root."""
    revs = [p for p in BUNDLE_ROOT.iterdir() if p.is_dir()]
    if not revs:
        raise SystemExit(f"no run bundles under {BUNDLE_ROOT}")
    return max(revs, key=lambda p: p.stat().st_mtime).name


def score_bundle(bundle: Path) -> pd.DataFrame:
    """Score one bundle's out-of-fold predictions with the official scorer."""
    frame = pd.read_parquet(bundle / "out_of_fold.parquet")
    predictions = frame[["Molecule_Name", *[f"{e}_pred" for e in REGRESSION_ENDPOINTS]]].rename(
        columns={f"{e}_pred": e for e in REGRESSION_ENDPOINTS}
    )
    truth_columns = ["Molecule_Name"]
    for endpoint in REGRESSION_ENDPOINTS:
        truth_columns += [endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]
    return score_submission(predictions, frame[truth_columns])


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=None, help="Revision namespace; default is the latest.")
    arguments = parser.parse_args()
    rev = arguments.rev or latest_rev()
    root = BUNDLE_ROOT / rev

    rows = []
    for bundle in sorted(root.iterdir()):
        match = _DIR.match(bundle.name)
        if not match or not (bundle / "out_of_fold.parquet").exists():
            continue
        scores = score_bundle(bundle)
        rows.append(
            {
                "phase": match["phase"],
                "name": match["name"],
                "seed": int(match["seed"]),
                "macro_spearman": scores.loc[MACRO_LABEL, "Spearman_R_mean"],
                "macro_st_rae": scores.loc[MACRO_LABEL, "ST-RAE_mean"],
                "macro_r2": scores.loc[MACRO_LABEL, "R2_mean"],
                **{
                    f"{e.split('_')[0]}_spearman": scores.loc[e, "Spearman_R_mean"]
                    for e in REGRESSION_ENDPOINTS
                },
            }
        )
    if not rows:
        raise SystemExit(f"no scored bundles under {root}")

    table = pd.DataFrame(rows)
    per_config = (
        table.drop(columns="seed")
        .groupby("name")
        .mean(numeric_only=True)
        .sort_values("macro_st_rae", ascending=True)
        .round(4)
    )
    seeds = table.groupby("name")["seed"].nunique()
    per_config.insert(0, "seeds", seeds)
    print(f"=== rev {rev}: configs ranked by macro ST-RAE (the challenge metric, lower better) ===")
    print("(Spearman/R2 are transfer diagnostics, not the objective)")
    print(per_config.to_string())


if __name__ == "__main__":
    main()
