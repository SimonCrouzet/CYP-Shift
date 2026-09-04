"""Fetching enzyme sequences from UniProt.

Two things downstream need a sequence rather than a structure. Scanning for the
conserved P450 signatures finds enzymes related closely enough to pretrain on, and
that has to work on candidates with no solved structure -- which is most of them.
Sequence-derived composition descriptors are the second: Lapins and Spjuth's
proteochemometric model of CYP inhibition (PLoS ONE 2013;8:e66566) covers our four
enzymes plus CYP2C19 and takes its target descriptors from the primary sequence alone.

Accessions are pinned rather than searched. A search for "CYP2C9 human" is not stable
across UniProt releases, and silently resolving to a different isoform would move every
descriptor without anything failing.
"""

import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
DEFAULT_SEQUENCE_DIR = Path("data/sequences")

#: Reviewed UniProt entries for the scored enzymes and the auxiliaries considered for
#: pretraining. Canonical sequence in each case.
ACCESSIONS: dict[str, str] = {
    "CYP1A2": "P05177",
    "CYP2C9": "P11712",
    "CYP2D6": "P10635",
    "CYP3A4": "P08684",
    "CYP2C19": "P33261",
    "CYP1A1": "P04798",
    "CYP1B1": "Q16678",
    "CYP2C8": "P10632",
    "CYP2B6": "P20813",
}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_fasta(text: str) -> str:
    """Sequence from a single-record FASTA, uppercased and stripped of gaps."""
    lines = [line.strip() for line in text.splitlines() if not line.startswith(">")]
    sequence = "".join(lines).upper().replace("-", "")
    if not sequence:
        raise ValueError("FASTA record contains no sequence")
    unknown = set(sequence) - set(AMINO_ACIDS)
    if unknown:
        logger.info("sequence contains non-standard residues: %s", sorted(unknown))
    return sequence


def fetch_sequence(accession: str, directory: Path = DEFAULT_SEQUENCE_DIR) -> str:
    """Download one UniProt entry, reusing a local copy when present."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{accession}.fasta"
    if not path.exists():
        logger.info("fetching %s from UniProt", accession)
        response = requests.get(UNIPROT_URL.format(accession=accession), timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    return parse_fasta(path.read_text())


def sequence_for(name: str, directory: Path = DEFAULT_SEQUENCE_DIR) -> str:
    """Sequence for a registered enzyme, by gene symbol.

    Raises:
        KeyError: If the enzyme has no pinned accession. Add one to
            :data:`ACCESSIONS` rather than searching, so the descriptor stays fixed.
    """
    if name not in ACCESSIONS:
        raise KeyError(f"{name} has no pinned UniProt accession")
    return fetch_sequence(ACCESSIONS[name], directory)


def read_fasta(path: Path) -> dict[str, str]:
    """Every record in a multi-record FASTA, keyed by the identifier on its header.

    The identifier is the first whitespace-delimited field after ``>``, which is what
    both UniProt and NCBI put there.
    """
    records: dict[str, str] = {}
    identifier: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if identifier is not None:
                records[identifier] = "".join(chunks).upper()
            identifier = re.split(r"\s+", line[1:].strip())[0]
            chunks = []
        elif identifier is not None:
            chunks.append(line.strip())
    if identifier is not None:
        records[identifier] = "".join(chunks).upper()
    logger.info("read %d sequences from %s", len(records), path.name)
    return records
