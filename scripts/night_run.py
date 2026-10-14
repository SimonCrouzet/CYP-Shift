"""Settle the regression config overnight, then write a submission.

The first sweep ranked configurations on a single holdout split, so its top gaps
(octant/tox21 versus the plain screen baseline, all within ~0.01) sit inside the
noise of one split. This run separates signal from noise and ends in a file to
submit:

* Phase A -- robustness. The handful of configurations that led the sweep, plus the
  two marginal winners combined (octant+tox21), each scored across five holdout
  seeds. A configuration only counts as better if it beats the baseline on the
  mean, not on one lucky split.
* Phase B -- tuning. The boosted hyperparameters are swept around whichever Phase A
  configuration won, across three seeds, so the submitted model is tuned rather than
  left at defaults.
* Phase C -- submission. The overall best (lowest mean macro ST-RAE) is refit on all
  training data and used to predict the blinded test set, written to
  ``submissions/``. The winning config is saved next to it.

Resilient like the sweep: every (phase, name, seed) result is appended to
``data/night_results.csv`` as it lands under a fixed schema, tagged with the code
revision. A re-run skips only rows completed at the *current* revision, so a code change
(e.g. the boosted weight fix) re-runs rather than reusing stale results; failures are
recorded and retried, not treated as done. Boosted only -- fast and the family that has
led every read so far.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/night_run.py
"""

import logging
import subprocess
import time
from pathlib import Path

import pandas as pd

from cypshift.cli import submit
from cypshift.eval.scoring import MACRO_LABEL
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import run

logger = logging.getLogger("night")

RESULTS = Path("data/night_results.csv")
BEST_CONFIG = Path("configs/night_best.json")
SUBMISSION = Path("submissions/night_submission_regression.csv")

OCT = {"sources.octant": True}
TOX = {"sources.tox21": True}
ALL_AUX = {
    "sources.sair": True,
    "sources.novartis": True,
    "sources.tox21": True,
    "sources.chembl": True,
    "sources.octant": True,
    "sources.veith": True,
}

#: The sweep's leaders plus the untried combination of its two marginal winners.
PHASE_A: dict[str, dict] = {
    "baseline": {},
    "octant": OCT,
    "tox21": TOX,
    "octant_tox21": {**OCT, **TOX},
    "octant_tox21_curriculum": {**OCT, **TOX, "training.preset": "coarse_to_fine"},
    "all_curriculum": {**ALL_AUX, "training.preset": "coarse_to_fine"},
}
PHASE_A_SEEDS = [0, 1, 2, 3, 4]

#: Boosted hyperparameters, swept around the Phase A winner (tag -> overrides).
HYPERPARAMETERS: dict[str, dict] = {
    "nl31": {"model.boosted.num_leaves": 31},
    "nl127": {"model.boosted.num_leaves": 127},
    "lr0.03": {"model.boosted.learning_rate": 0.03},
    "ne1000": {"model.boosted.n_estimators": 1000},
    "nl127_ne1000": {"model.boosted.num_leaves": 127, "model.boosted.n_estimators": 1000},
    "nl127_lr0.03": {"model.boosted.num_leaves": 127, "model.boosted.learning_rate": 0.03},
    "deep_slow": {
        "model.boosted.num_leaves": 127,
        "model.boosted.learning_rate": 0.03,
        "model.boosted.n_estimators": 1000,
    },
}
PHASE_B_SEEDS = [0, 1, 2]


#: One fixed row schema, so a failure row cannot have fewer fields than a success row
#: and push later rows under the wrong columns.
RESULT_COLUMNS = [
    "rev", "phase", "name", "seed", "status", "macro_ST_RAE", "minutes",
    "CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4",
]  # fmt: skip


def run_identity() -> str:
    """A code-revision tag (short SHA, ``-dirty`` when uncommitted), or ``nogit``.

    Results are namespaced by it so a code change (e.g. the boosted weight fix) does not
    silently reuse results trained under the old semantics.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        clean = subprocess.call(["git", "diff", "--quiet"]) == 0
        clean = clean and subprocess.call(["git", "diff", "--cached", "--quiet"]) == 0
        return sha if clean else f"{sha}-dirty"
    except Exception:  # noqa: BLE001 -- no git is not fatal, just a weaker identity
        return "nogit"


REV = run_identity()


def already_done() -> set[tuple[str, str, int]]:
    """Completed (phase, name, seed) at the current revision; failures are retried."""
    if not RESULTS.exists():
        return set()
    table = pd.read_csv(RESULTS)
    if "rev" not in table.columns or "status" not in table.columns:
        return set()  # pre-schema results from an earlier code era: re-run under this one
    ok = table[(table["rev"] == REV) & (table["status"] == "ok")]
    return {(r.phase, r.name, int(r.seed)) for r in ok.itertuples()}


def record(row: dict) -> None:
    full = pd.DataFrame([{column: row.get(column) for column in RESULT_COLUMNS}])
    full.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def evaluate(phase: str, name: str, overrides: dict, seed: int) -> None:
    config = ExperimentConfig.load()
    config.override("data.seed", seed)
    for key, value in overrides.items():
        config.override(key, value)
    config.validate()
    started = time.time()
    scores, _ = run(config)
    macro = float(scores.loc[MACRO_LABEL, "ST-RAE_mean"])
    row = {
        "rev": REV,
        "phase": phase,
        "name": name,
        "seed": seed,
        "status": "ok",
        "macro_ST_RAE": round(macro, 4),
        "minutes": round((time.time() - started) / 60, 1),
    }
    for endpoint, data in scores.iterrows():
        if endpoint != MACRO_LABEL:
            row[endpoint.split("_")[0]] = round(float(data["ST-RAE_mean"]), 4)
    record(row)
    logger.info(
        ">>> %s/%s seed=%d: macro %.4f (%.1f min)", phase, name, seed, macro, row["minutes"]
    )


def run_phase(phase: str, configs: dict[str, dict], seeds: list[int]) -> None:
    done = already_done()
    for name, overrides in configs.items():
        for seed in seeds:
            if (phase, name, seed) in done:
                logger.info("skip %s/%s seed=%d (already recorded)", phase, name, seed)
                continue
            try:
                evaluate(phase, name, overrides, seed)
            except Exception as error:  # noqa: BLE001 -- one failure must not sink the night
                logger.exception("%s/%s seed=%d FAILED: %s", phase, name, seed, error)
                record({"rev": REV, "phase": phase, "name": name, "seed": seed, "status": "failed"})


def _completed(phase: str | None = None) -> pd.DataFrame:
    """Successful rows at the current revision, optionally for one phase."""
    table = pd.read_csv(RESULTS)
    table = table[(table["rev"] == REV) & (table["status"] == "ok")].copy()
    table["macro_ST_RAE"] = table["macro_ST_RAE"].astype(float)
    return table if phase is None else table[table["phase"] == phase]


def ranked_means(phase: str | None = None) -> pd.DataFrame:
    """Mean and spread of macro ST-RAE per configuration, best first."""
    return (
        _completed(phase)
        .groupby("name")["macro_ST_RAE"]
        .agg(["mean", "std", "count"])
        .sort_values("mean")
    )


def ranked_on_common_seeds(names: set[str]) -> pd.DataFrame:
    """Rank the named configurations on the seeds every one of them completed.

    Comparing candidates evaluated on different seed sets is unfair; restrict to the seeds
    they all share so the means are paired.
    """
    table = _completed()
    table = table[table["name"].isin(names)]
    if table.empty:
        return table
    seeds_per_name = table.groupby("name")["seed"].agg(set)
    common = set.intersection(*seeds_per_name) if len(seeds_per_name) else set()
    if common:
        table = table[table["seed"].isin(common)]
    return table.groupby("name")["macro_ST_RAE"].agg(["mean", "std", "count"]).sort_values("mean")


def _archive_stale_results() -> None:
    """Move an old-schema results file aside so a fresh namespace starts clean."""
    if RESULTS.exists() and "rev" not in pd.read_csv(RESULTS, nrows=0).columns:
        archived = RESULTS.with_suffix(".pre_schema.csv")
        RESULTS.rename(archived)
        logger.info("archived pre-schema results to %s", archived)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _archive_stale_results()
    logger.info("run identity (results namespace): %s", REV)

    logger.info("=== Phase A: robustness across %d seeds ===", len(PHASE_A_SEEDS))
    run_phase("A", PHASE_A, PHASE_A_SEEDS)
    phase_a = ranked_means("A")
    logger.info("\nPhase A (mean macro ST-RAE, lower better):\n%s", phase_a.to_string())
    winner = phase_a.index[0]
    winner_overrides = PHASE_A[winner]
    logger.info("Phase A winner: %s (mean %.4f)", winner, phase_a.loc[winner, "mean"])

    logger.info("=== Phase B: tuning boosted hyperparameters around %s ===", winner)
    phase_b_configs = {
        f"{winner}+{tag}": {**winner_overrides, **hp} for tag, hp in HYPERPARAMETERS.items()
    }
    registry = {winner: winner_overrides, **phase_b_configs}
    run_phase("B", phase_b_configs, PHASE_B_SEEDS)

    # Overall best across both phases, compared on the seeds the candidates share so the
    # winner (scored on more seeds) and the tuned variants (fewer) are ranked on equal
    # footing rather than on whichever seed set each happened to complete.
    candidates = ranked_on_common_seeds(set(registry))
    logger.info("\nCandidates (mean macro ST-RAE, common seeds):\n%s", candidates.to_string())
    best = candidates.index[0]
    logger.info("Overall best: %s (mean %.4f)", best, candidates.loc[best, "mean"])

    logger.info("=== Phase C: refit %s on all data and write the submission ===", best)
    config = ExperimentConfig.load()
    for key, value in registry[best].items():
        config.override(key, value)
    config.validate()
    config.save(BEST_CONFIG)
    submit(output=SUBMISSION, config_path=BEST_CONFIG, track="regression")
    logger.info("submission written to %s (config %s)", SUBMISSION, BEST_CONFIG)


if __name__ == "__main__":
    main()
