#!/usr/bin/env python
"""Run one configured experiment. This is what a sweep agent invokes.

    python scripts/run_experiment.py --config configs/default.json
    python scripts/run_experiment.py --set model.kind=neural conditioning.strategy=flat

Any configuration field can be overridden with ``--set section.field=value``. A Weights
& Biases sweep's default command instead passes each parameter as a bare long flag
(``--section.field=value``); both forms are accepted, so the sweep configs need only a
command block that adds ``--track``.
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
    """Read a value the way JSON would, with Python-style booleans, else the plain string."""
    stripped = text.strip()
    if stripped in ("True", "False"):  # W&B serialises booleans Python-style
        return stripped == "True"
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def overrides_from(set_args: list[str] | None, extras: list[str]) -> list[str]:
    """Collect ``section.field=value`` assignments from ``--set`` and from W&B-style flags.

    A W&B sweep's default command passes each parameter as ``--section.field=value``, not
    ``--set section.field=value``, so the bare long flags are accepted as well.
    """
    assignments = list(set_args or [])
    for item in extras:
        if item.startswith("--") and "=" in item:
            assignments.append(item[2:])
        else:
            raise SystemExit(f"unrecognised argument {item!r}; expected --section.field=value")
    return assignments


def build_config(arguments: argparse.Namespace, extras: list[str]) -> ExperimentConfig:
    config = ExperimentConfig.load(Path(arguments.config))
    for assignment in overrides_from(arguments.set, extras):
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
    arguments, extras = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.WARNING if arguments.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = build_config(arguments, extras)
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
