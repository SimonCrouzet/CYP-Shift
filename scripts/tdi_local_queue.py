"""Run the TDI jobs on this machine, one at a time.

    python scripts/tdi_local_queue.py --list
    python scripts/tdi_local_queue.py --only lightgbm,knn
    python scripts/tdi_local_queue.py --dry-run

Each job is a command with the files it writes. A job whose files all exist is skipped (``--force``
runs it again), so the queue can be stopped and started again. Jobs run in the order below at low
priority (``nice``), each with its log in ``data/tdi_runs/queue_logs/``; a failed job is reported at
the end and the queue goes on with the next.

Stages: ``lightgbm`` (three feature sets, with test scores), ``knn`` (Tanimoto neighbours),
``tabpfn`` (a one-fold timing run on the CheMeleon + dense set, 128 components; its time says
whether the full run fits the 8 GB GPU), ``noise`` (LightGBM on the CheMeleon + dense set with the
labels as given, without the ones whose measured shift is within one standard error of the 2-fold
line, down-weighted by that distance, and replaced by the measured shift), ``novartis`` (the graph
cell below with the Novartis CYP3A4 log_kobs as a response task, on its first two folds),
``graph`` (seeds 1 and 2 of the TDI-fold graph cell with the label rows of compounds without a
direct curve at weight 0). The graph stages take hours each on this GPU.
"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("tdi_local_queue")

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("data/tdi_runs")
LOGS = RUNS / "queue_logs"
BASELINE = "scripts/tdi_baseline.py"
STAGES = ("lightgbm", "knn", "tabpfn", "noise", "novartis", "graph")
#: The graph cell whose seeds are run, and the name of its recipe in the bundles.
GRAPH_CELL = "tdi5_label_b_reg_uncovered_0"
GRAPH_NAME = "focus_tdi5_label_b_reg_uncovered_0_8352aa"
GRAPH_REV = "questions3"
GRAPH_SEEDS = (1, 2)
#: The same cell with the Novartis CYP3A4 log_kobs as a response task (seed 0, to pair with the
#: seed-0 run of the cell above).
KOBS_CELL = "tdi5_label_b_reg_uncovered_0_kobs"
KOBS_NAME = "focus_tdi5_label_b_reg_uncovered_0_kobs_ab22d4"
#: Its first two folds only: the paired comparison with the seed-0 run is made on those folds.
KOBS_FOLDS = (0, 1)


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    env: dict[str, str] = field(default_factory=dict)


def baseline_job(name: str, stage: str, *options: str, test: bool = True) -> Job:
    """A ``tdi_baseline.py`` run trained on the directly measured compounds, writing its scores to
    ``f5_<name>.csv`` and, with ``test``, the test scores to ``f5_<name>_test.csv``."""
    out, test_out = RUNS / f"f5_{name}.csv", RUNS / f"f5_{name}_test.csv"
    argv = [BASELINE, *options, "--train-on", "covered", "--threads", "4", "--output", str(out)]
    if test:
        argv += ["--test-output", str(test_out)]
    return Job(name, stage, tuple(argv), (test_out if test else out,))


def graph_job(
    seed: int,
    cell: str = GRAPH_CELL,
    name: str = GRAPH_NAME,
    stage: str = "graph",
    folds: tuple[int, ...] = (),
) -> Job:
    """One seed of a graph cell, run through ``night_run.py`` as the Modal runs are. With ``folds``
    only those folds are run, and the bundle is named for them."""
    suffix = f"_folds{''.join(map(str, folds))}" if folds else ""
    bundle = Path("data/run_bundles") / f"{GRAPH_REV}-m1" / f"reg_{name}{suffix}_seed{seed}"
    env = {
        "CYP_PLAN": "questions",
        "CYP_CELLS": cell,
        "CYP_FOLDS": ",".join(map(str, folds)),
        "CYP_SEEDS": str(seed),
        "CYP_REV": GRAPH_REV,
        "CYP_MEMBERS": "1",
        "CYP_JOBS": "1",
        "CYP_WORKERS": "0",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}",
    }
    return Job(
        f"{stage}_seed{seed}" if stage == "graph" else f"{cell}{suffix}_seed{seed}",
        stage,
        ("scripts/night_run.py",),
        (bundle / "out_of_fold_auxiliary.parquet",),
        env,
    )


def jobs() -> list[Job]:
    return [
        baseline_job("lgb_fp_dense_covered", "lightgbm", "--features", "fp_dense"),
        baseline_job("lgb_chemeleon_covered", "lightgbm", "--features", "chemeleon"),
        baseline_job("lgb_dense_chemeleon_covered", "lightgbm", "--features", "dense_chemeleon"),
        baseline_job("knn_fp_dense_covered", "knn", "--model", "knn", "--features", "fp_dense"),
        baseline_job(
            "tabpfn_dense_chemeleon_pca128_fold1",
            "tabpfn",
            *("--model", "tabpfn", "--features", "dense_chemeleon", "--pca", "128"),
            *("--max-folds", "1"),
            test=False,
        ),
        *(
            baseline_job(
                f"lgb_dense_chemeleon_{mode}",
                "noise",
                *("--features", "dense_chemeleon", "--label-mode", mode),
                test=False,
            )
            for mode in ("hard", "drop_ambiguous", "soft", "shift")
        ),
        graph_job(0, KOBS_CELL, KOBS_NAME, "novartis", KOBS_FOLDS),
        *(graph_job(seed) for seed in GRAPH_SEEDS),
    ]


def selected(only: str | None) -> list[Job]:
    """The jobs of the named stages (comma-separated; all when ``only`` is empty), in order."""
    stages = only.split(",") if only else list(STAGES)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage {sorted(unknown)}; stages are {STAGES}")
    return [job for job in jobs() if job.stage in stages]


def command(job: Job) -> list[str]:
    return ["nice", "-n", "10", sys.executable, *job.argv]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", help=f"comma-separated stages of {STAGES}")
    parser.add_argument(
        "--list", action="store_true", help="show the jobs and whether they are done"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--force", action="store_true", help="run jobs whose files exist as well")
    args = parser.parse_args()

    queue = selected(args.only)
    done = {job.name: all((ROOT / path).exists() for path in job.outputs) for job in queue}
    if args.list:
        for job in queue:
            logger.info("%-8s %-40s %s", job.stage, job.name, "done" if done[job.name] else "todo")
        return
    failed = []
    for job in queue:
        if done[job.name] and not args.force:
            logger.info("skip %s: its files exist", job.name)
            continue
        logger.info("%s: %s", job.name, " ".join(command(job)))
        if args.dry_run:
            continue
        (ROOT / LOGS).mkdir(parents=True, exist_ok=True)
        with (ROOT / LOGS / f"{job.name}.log").open("w") as log:
            result = subprocess.run(  # noqa: S603 -- fixed commands of this repo
                command(job),
                cwd=ROOT,
                env={**os.environ, **job.env},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        logger.info("%s: exit %d", job.name, result.returncode)
        if result.returncode:
            failed.append(job.name)
    if failed:
        logger.info("failed: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
