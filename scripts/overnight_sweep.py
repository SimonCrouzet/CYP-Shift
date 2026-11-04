"""Run a focused configuration sweep on the similarity-matched holdout, overnight.

Answers the questions the codebase left to "let the sweep decide": which auxiliary
sources help, whether the per-assay calibration beats a plain one-hot, whether the
training curriculum beats pooling, which objective and model family win. Nothing is
submitted -- this only scores configurations on the local holdout and records the macro
soft-threshold RAE.

Resilient by design: each configuration is wrapped, so one failure does not sink the run;
results are appended to ``data/sweep_results.csv`` as they land and a re-run skips
configurations already recorded there. Boosted (fast) configurations run first, so even a
partial night yields the source answer; the slower neural/curve ones run last.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/overnight_sweep.py
"""

import logging
import time
from pathlib import Path

import pandas as pd

from cypshift.eval.scoring import MACRO_LABEL
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import run

logger = logging.getLogger("sweep")

RESULTS = Path("data/sweep_results.csv")
ALL_AUX = {
    "sources.sair": True,
    "sources.novartis": True,
    "sources.tox21": True,
    "sources.chembl": True,
    "sources.octant": True,
    "sources.veith": True,
}
NEAR_AUX = {"sources.sair": True, "sources.novartis": True}

#: (name, overrides). Boosted first (fast, most informative), neural/curve last (slow).
GRID: list[tuple[str, dict]] = [
    ("boosted_baseline", {}),
    ("boosted_sair", {"sources.sair": True}),
    ("boosted_novartis", {"sources.novartis": True}),
    ("boosted_tox21", {"sources.tox21": True}),
    ("boosted_veith", {"sources.veith": True}),
    ("boosted_chembl", {"sources.chembl": True}),
    ("boosted_octant", {"sources.octant": True}),
    ("boosted_all_aux", ALL_AUX),
    ("boosted_all_dist0", {**ALL_AUX, "auxiliary.max_distance": 0.0}),
    ("boosted_all_dist3", {**ALL_AUX, "auxiliary.max_distance": 3.0}),
    ("boosted_all_sw0.3", {**ALL_AUX, "auxiliary.scored_weight": 0.3}),
    ("boosted_all_absolute", {**ALL_AUX, "model.objective": "absolute"}),
    ("boosted_all_2048", {**ALL_AUX, "features.fingerprint_bits": 2048}),
    ("boosted_all_curriculum", {**ALL_AUX, "training.preset": "coarse_to_fine"}),
    ("curve_baseline", {"model.kind": "curve"}),
    ("curve_near_aux", {"model.kind": "curve", **NEAR_AUX}),
    (
        "neural_factorised",
        {"model.kind": "neural", "conditioning.strategy": "factorised", **NEAR_AUX},
    ),
    (
        "neural_calibration",
        {"model.kind": "neural", "conditioning.strategy": "calibration", **NEAR_AUX},
    ),
    (
        "neural_curriculum",
        {
            "model.kind": "neural",
            "conditioning.strategy": "heads",
            "training.preset": "coarse_to_fine",
            **NEAR_AUX,
        },
    ),
]


def already_done() -> set[str]:
    if not RESULTS.exists():
        return set()
    table = pd.read_csv(RESULTS)
    # Exclude FAILED rows so a failed config is retried on the next run rather than skipped
    # as "done" by name.
    return set(table[table["macro_ST_RAE"].astype(str) != "FAILED"]["name"])


def record(row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


def evaluate(name: str, overrides: dict) -> None:
    config = ExperimentConfig.load()
    for key, value in overrides.items():
        config.override(key, value)
    config.validate()
    started = time.time()
    scores, _ = run(config)
    macro = float(scores.loc[MACRO_LABEL, "ST-RAE_mean"])
    row = {
        "name": name,
        "macro_ST_RAE": round(macro, 4),
        "minutes": round((time.time() - started) / 60, 1),
    }
    for endpoint, data in scores.iterrows():
        if endpoint != MACRO_LABEL:
            row[endpoint.split("_")[0]] = round(float(data["ST-RAE_mean"]), 4)
    record(row)
    logger.info(">>> %s: macro ST-RAE %.4f (%.1f min)", name, macro, row["minutes"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = already_done()
    pending = [(n, o) for n, o in GRID if n not in done]
    logger.info("sweep: %d configs, %d already done, %d to run", len(GRID), len(done), len(pending))
    for name, overrides in pending:
        logger.info("--- running %s ---", name)
        try:
            evaluate(name, overrides)
        except Exception as error:  # noqa: BLE001 -- one failure must not sink the night
            logger.exception("%s FAILED: %s", name, error)
            record({"name": name, "macro_ST_RAE": "FAILED", "minutes": 0})

    if RESULTS.exists():
        table = pd.read_csv(RESULTS)
        ranked = table[table["macro_ST_RAE"] != "FAILED"].copy()
        ranked["macro_ST_RAE"] = ranked["macro_ST_RAE"].astype(float)
        ranked = ranked.sort_values("macro_ST_RAE")
        logger.info("\n=== LEADERBOARD (lower is better) ===\n%s", ranked.to_string(index=False))


if __name__ == "__main__":
    main()
