"""Scan a FASTA for cytochrome P450s worth pretraining on.

The pretraining set is a lever we control. Auxiliary enzymes contribute data only if
they share the fold and the recognition chemistry, and a candidate normally arrives as
a sequence with no solved structure -- so the pocket descriptors cannot be computed for
it until it has earned the effort of finding one. This filters candidates first.

The conserved signatures it looks for are useless for telling two P450s apart, since
they are invariant across the superfamily. That is precisely why they work here: the
question is membership, not identity.

Usage::

    python scripts/find_related_enzymes.py proteome.fasta
    python scripts/find_related_enzymes.py proteome.fasta --threshold 0.9 --explain
"""

import argparse
import logging
import sys
from pathlib import Path

from cypshift.data.motifs import describe, p450_score, scan
from cypshift.data.sequences import read_fasta

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path, help="FASTA file of candidate sequences")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="minimum signature score to report (default: %(default)s)",
    )
    parser.add_argument(
        "--explain", action="store_true", help="show which signatures each hit carried"
    )
    parser.add_argument("--output", type=Path, help="write accepted identifiers here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = read_fasta(args.fasta)
    scored = sorted(
        ((name, seq, p450_score(seq)) for name, seq in records.items()),
        key=lambda row: row[2],
        reverse=True,
    )
    accepted = [(name, seq, score) for name, seq, score in scored if score >= args.threshold]

    print(f"{len(accepted)} of {len(records)} sequences pass at threshold {args.threshold}")
    for name, sequence, score in accepted:
        found = scan(sequence)
        carried = ",".join(m for m, p in found.positions.items() if p is not None) or "-"
        print(f"  {score:.2f}  {name:<30} {len(sequence):>5} aa  heme+{carried}")
        if args.explain:
            print("\n".join(f"      {line}" for line in describe(sequence).splitlines()))

    if args.output:
        args.output.write_text("\n".join(name for name, _, _ in accepted) + "\n")
        logger.info("wrote %d identifiers to %s", len(accepted), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
