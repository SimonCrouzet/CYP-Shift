#!/usr/bin/env python
"""Score a configured model on the analogue view, the second validation lens the note asks for.

Scaffold CV and the similarity-matched holdout both reproduce the test set's *distance* but
not its *construction*: analogues bought around measured reference hits, scored while the
anchor stays known. :func:`cypshift.data.splits.analogue_holdout` builds that view -- retained
anchors, their structural neighbours held out -- and this runs the usual pipeline on it, so a
calibration or local-correction change can be read where it is meant to bite. Scoring drops
unmeasured endpoints, so only genuinely observed labels are judged.

    python scripts/analogue_validation.py --set model.kind=graph --holdout-frac 0.1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.data.challenge import load
from cypshift.data.splits import analogue_split
from cypshift.experiment import DEFAULT_CONFIG_PATH, ExperimentConfig
from cypshift.pipeline import run

logger = logging.getLogger("cypshift.analogue_validation")


def parse_value(text: str) -> object:
    stripped = text.strip()
    if stripped in ("True", "False"):
        return stripped == "True"
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--set", nargs="*", metavar="section.field=value", default=[])
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--min-similarity", type=float, default=0.4)
    parser.add_argument("--bundle", type=Path, default=None, help="Write a run bundle here.")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if arguments.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = ExperimentConfig.load(Path(arguments.config))
    for assignment in arguments.set:
        dotted, _, raw = assignment.partition("=")
        config.override(dotted.strip(), parse_value(raw.strip()))
    config.validate()

    # The mask must line up with the row order run() loads the training set in, so it is built
    # from the same load.
    train = load("inhibition", Path(config.data.directory))
    labels = train[list(REGRESSION_ENDPOINTS)].notna().to_numpy()
    split = analogue_split(
        train["SMILES"].tolist(),
        holdout_fraction=arguments.holdout_frac,
        min_similarity=arguments.min_similarity,
        seed=config.data.seed,
    )
    # Report whether each held-out compound actually has a same-endpoint retained anchor: the
    # mask alone does not guarantee it, and a local corrector depends on it.
    coverage = split.anchor_label_coverage(labels)
    print(
        "same-endpoint anchor coverage:",
        {
            REGRESSION_ENDPOINTS[c].split("_")[0]: (None if v != v else round(v, 3))
            for c, v in coverage.items()
        },
    )
    scores, _ = run(config, bundle_path=arguments.bundle, holdout_mask=split.holdout_mask)
    print(scores.round(4).to_string())
    print(f"\nanalogue-view MA-ST-RAE: {scores.loc['MA', 'ST-RAE_mean']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
