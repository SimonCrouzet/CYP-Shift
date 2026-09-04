#!/usr/bin/env python
"""Run one configured experiment. This is what a sweep agent invokes.

    python scripts/run_experiment.py --config configs/default.json
    python scripts/run_experiment.py --set model.kind=neural conditioning.strategy=flat

Any configuration field can be overridden with ``--set section.field=value``, which is
also the form a Weights & Biases sweep passes on the command line.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from cypshift.experiment import DEFAULT_CONFIG_PATH, ExperimentConfig
from cypshift.pipeline import metrics_for_tracking, run
from cypshift.tracking import track

logger = logging.getLogger("cypshift.run")


def parse_value(text: str) -> object:
    """Read a value the way JSON would, falling back to the plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(arguments: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.load(Path(arguments.config))
    for assignment in arguments.set or []:
        if "=" not in assignment:
            raise SystemExit(f"expected section.field=value, got {assignment!r}")
        dotted, _, raw = assignment.partition("=")
        config.override(dotted.strip(), parse_value(raw.strip()))
    if arguments.track:
        config.override("tracking.enabled", True)
    return config.validate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--set", nargs="*", metavar="section.field=value")
    parser.add_argument("--track", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument("--save-config", type=Path, help="Write the resolved config here.")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if arguments.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = build_config(arguments)
    if arguments.save_config:
        config.save(arguments.save_config)

    with track(config) as tracker:
        scores, _ = run(config)
        tracker.log(metrics_for_tracking(scores), prefix="holdout")

    print(scores.round(4).to_string())
    print(f"\nMA-ST-RAE: {scores.loc['MA', 'ST-RAE_mean']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
