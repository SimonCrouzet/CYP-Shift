"""Rank enzymes as pretraining candidates by binding-site similarity and data volume.

An auxiliary enzyme is worth training on only if both hold: its binding site resembles
a scored enzyme's, so what is learned transfers, and public data exists for it, so
there is something to learn. The two turn out to run opposite to each other among human
CYPs -- CYP3A4's three closest relatives have almost no measured inhibition between
them -- so neither criterion alone gives a usable answer.

Similarity is measured over the substrate recognition sites rather than the whole
chain, because the P450 scaffold is invariant and inflates any global comparison
without saying whether two enzymes prefer the same ligands. See
:mod:`cypshift.data.site`.

Usage::

    python scripts/rank_pretraining_candidates.py --fasta human_cyps.fasta
    python scripts/rank_pretraining_candidates.py --fasta cyps.fasta --no-chembl
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests

from cypshift.config import ISOFORMS
from cypshift.data.motifs import p450_score
from cypshift.data.sequences import sequence_for
from cypshift.data.site import site_distance

logger = logging.getLogger(__name__)

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=reviewed:true+AND+organism_id:9606+AND+gene:CYP*&format=fasta"
)


def read_named_fasta(path: Path) -> dict[str, str]:
    """Sequences keyed by gene symbol, taken from the ``GN=`` field of the header."""
    sequences: dict[str, list[str]] = {}
    gene: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            match = re.search(r"GN=(\S+)", line)
            gene = match.group(1) if match else line[1:].split()[0]
            sequences[gene] = []
        elif gene:
            sequences[gene].append(line.strip())
    return {gene: "".join(parts) for gene, parts in sequences.items() if parts}


def fetch_human_cyps(path: Path) -> dict[str, str]:
    """Download every reviewed human CYP-named entry, reusing a local copy."""
    if not path.exists():
        logger.info("fetching reviewed human CYP entries from UniProt")
        response = requests.get(UNIPROT_STREAM, timeout=300)
        response.raise_for_status()
        path.write_text(response.text)
    return read_named_fasta(path)


def chembl_counts(accession: str) -> tuple[str, int]:
    """Target identifier and IC50/Ki record count for one UniProt accession.

    Returns ``("-", 0)`` when the target is absent or the service does not answer.
    Missing data is reported as missing rather than raising, since one unavailable
    lookup should not sink the whole ranking.
    """
    try:
        targets = requests.get(
            f"{CHEMBL}/target.json",
            params={"target_components__accession": accession, "limit": 5},
            timeout=60,
        ).json()
        single = [
            t["target_chembl_id"]
            for t in targets.get("targets", [])
            if t.get("target_type") == "SINGLE PROTEIN"
        ]
        if not single:
            return "-", 0
        counted = requests.get(
            f"{CHEMBL}/activity.json",
            params={
                "target_chembl_id": single[0],
                "standard_type__in": "IC50,Ki",
                "limit": 1,
            },
            timeout=120,
        ).json()
        return single[0], int(counted["page_meta"]["total_count"])
    except (requests.RequestException, KeyError, ValueError) as error:
        logger.info("ChEMBL lookup failed for %s: %s", accession, error)
        return "-", 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/sequences/human_cyps.fasta"),
        help="candidate sequences; downloaded from UniProt if absent",
    )
    parser.add_argument("--no-chembl", action="store_true", help="skip the data-volume lookup")
    parser.add_argument("--top", type=int, default=15, help="rows to show")
    parser.add_argument("--output", type=Path, help="write the full table here as CSV")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.fasta.parent.mkdir(parents=True, exist_ok=True)
    candidates = fetch_human_cyps(args.fasta)

    scored = tuple(name.split("_")[0] for name in ISOFORMS)
    rejected = [g for g, s in candidates.items() if p450_score(s) < 0.7]
    logger.info("%d entries, %d rejected by the P450 signature", len(candidates), len(rejected))

    rows = []
    for gene, sequence in candidates.items():
        if gene in scored or p450_score(sequence) < 0.7:
            continue
        entry: dict[str, object] = {"gene": gene}
        try:
            for target in scored:
                entry[target] = round(site_distance(sequence_for(target), sequence), 2)
        except ValueError as error:
            logger.info("skipping %s: %s", gene, error)
            continue
        entry["closest"] = min(float(entry[t]) for t in scored)
        entry["nearest_target"] = min(scored, key=lambda t: float(entry[t]))
        rows.append(entry)

    table = pd.DataFrame(rows).sort_values("closest").set_index("gene")

    if not args.no_chembl:
        from cypshift.data.sequences import ACCESSIONS

        counts = []
        for gene in table.index:
            accession = ACCESSIONS.get(gene)
            counts.append(chembl_counts(accession)[1] if accession else None)
        table["chembl_ic50_ki"] = counts

    pd.set_option("display.width", 200)
    print(table.head(args.top).to_string())
    if args.output:
        table.to_csv(args.output)
        logger.info("wrote %d rows to %s", len(table), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
