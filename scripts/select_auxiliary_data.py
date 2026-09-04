"""Select auxiliary enzymes by binding-site distance and weight their data by it.

Two decisions, one number. Which enzymes to admit, and how much each one's rows should
count once admitted. Both follow from how far its binding site is from a scored
enzyme's, because that is what says whether a structure-activity relationship measured
on it carries over.

The threshold is on a scale where two unrelated binding sites score 5.27 and a
conservative Leu-to-Ile swap scores 0.62, so 1.5 is under a third of the random
baseline. Raising it admits enzymes fast: 1.5 gives seven among human CYPs, 3.0 gives
seventeen.

Usage::

    python scripts/select_auxiliary_data.py
    python scripts/select_auxiliary_data.py --max-distance 3.0 --weighting inverse
    python scripts/select_auxiliary_data.py --output data/auxiliary_weights.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from cypshift.config import ISOFORMS
from cypshift.data.auxiliary import (
    AUXILIARY_WEIGHTING,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_SCALE,
    select_auxiliaries,
    site_distances,
)
from cypshift.data.sequences import ACCESSIONS, fetch_sequence

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, help="candidate sequences; defaults to the registry")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--weighting", choices=AUXILIARY_WEIGHTING, default="exponential")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--output", type=Path, help="write the admitted enzymes here as CSV")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scored = tuple(name.split("_")[0] for name in ISOFORMS)

    if args.fasta:
        from scripts.rank_pretraining_candidates import read_named_fasta

        candidates = read_named_fasta(args.fasta)
    else:
        candidates = {name: fetch_sequence(acc) for name, acc in ACCESSIONS.items()}

    distances = site_distances(scored, candidates)
    admitted = select_auxiliaries(distances, args.max_distance, args.weighting, args.scale)

    if not admitted:
        print(f"nothing within {args.max_distance:.2f} of any scored enzyme")
        return 0

    table = pd.DataFrame(
        [
            {
                "enzyme": a.name,
                "nearest": a.nearest,
                "distance": round(a.distance, 3),
                "weight": round(a.weight, 3),
            }
            for a in admitted
        ]
    ).set_index("enzyme")
    print(table.to_string())

    covered = {a.nearest for a in admitted}
    for target in scored:
        if target not in covered:
            print(f"\n{target}: no auxiliary within {args.max_distance:.2f} -- trains alone")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
        logger.info("wrote %d enzymes to %s", len(table), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
