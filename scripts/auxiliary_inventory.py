"""Write the per-group inventory of the auxiliary rows a configuration trains on.

    python scripts/auxiliary_inventory.py --output data/inventory_full.csv \\
        --set model.kind=graph --set sources.octant=true --set sources.tox21=true
    python scripts/auxiliary_inventory.py --output data/inventory_random.csv ... \\
        --set auxiliary.row_selection=random --set auxiliary.max_rows_per_group=2000

``--set`` takes the same keys as the sweep's overrides; values are read as JSON, else as text.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from cypshift.data.challenge import load
from cypshift.data.inventory import auxiliary_inventory
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import auxiliary_frames


def _parse(pair: str) -> tuple[str, object]:
    key, _, text = pair.partition("=")
    try:
        return key, json.loads(text)
    except json.JSONDecodeError:
        return key, text


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--cluster-distance", type=float, default=0.6)
    args = parser.parse_args()

    config = ExperimentConfig.load()
    for pair in args.overrides:
        config.override(*_parse(pair))
    config.validate()
    table, response = auxiliary_frames(config, load("inhibition", args.directory))
    table = pd.concat([table, response], ignore_index=True)
    inventory = auxiliary_inventory(table, args.cluster_distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)
    print(inventory.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
