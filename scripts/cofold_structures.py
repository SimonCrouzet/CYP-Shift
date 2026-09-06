"""Precompute Boltz-2 cofold features for (enzyme, compound) pairs into the cache.

The GPU step behind the structure track. Run it in the ``cypshift-struct`` environment;
the pipeline then reads the cache with no GPU. Cofolding is expensive, so this is the
one place it happens, and re-runs skip pairs already cached.

Usage (in cypshift-struct)::

    python scripts/cofold_structures.py --enzymes CYP1A2 CYP2C9 --limit 50
    python scripts/cofold_structures.py --smiles-file mols.txt --enzymes CYP3A4
"""

import argparse
import logging
import sys
from pathlib import Path

from cypshift.config import ISOFORMS
from cypshift.data.challenge import load
from cypshift.structure.cofold import DEFAULT_CACHE, canonical_key, cofold_features

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enzymes", nargs="+", default=[n.split("_")[0] for n in ISOFORMS])
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--smiles-file", type=Path, help="One SMILES per line; defaults to the training set."
    )
    parser.add_argument("--limit", type=int, help="Cofold at most this many compounds per enzyme.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-affinity", action="store_true", help="Skip the affinity head.")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=200)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.smiles_file:
        smiles = [
            line.strip() for line in args.smiles_file.read_text().splitlines() if line.strip()
        ]
    else:
        smiles = load("inhibition", args.directory)["SMILES"].tolist()
    if args.limit:
        smiles = smiles[: args.limit]

    total = len(args.enzymes) * len(smiles)
    logger.info("cofolding up to %d (enzyme, compound) pairs", total)
    done = failed = 0
    for enzyme in args.enzymes:
        for compound in smiles:
            cached = args.cache / enzyme / f"{canonical_key(compound)}.json"
            if cached.exists():
                done += 1
                continue
            try:
                cofold_features(
                    enzyme,
                    compound,
                    cache=args.cache,
                    affinity=not args.no_affinity,
                    extra=(
                        "--recycling_steps",
                        str(args.recycling_steps),
                        "--sampling_steps",
                        str(args.sampling_steps),
                    ),
                )
                done += 1
            except Exception as error:  # noqa: BLE001 -- keep going; a failed pair is NaN
                failed += 1
                logger.warning("cofold failed for %s + %s: %s", enzyme, compound[:40], error)
            if (done + failed) % 10 == 0:
                logger.info("progress: %d done, %d failed of %d", done, failed, total)
    logger.info("finished: %d cofolded, %d failed", done, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
