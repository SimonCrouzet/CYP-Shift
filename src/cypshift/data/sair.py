"""Auxiliary enzyme activity from SAIR, SandboxAQ's cofolded IC50 repository.

SAIR (``huggingface.co/datasets/SandboxAQ/SAIR``, CC-BY-4.0) is ~1M protein-ligand
pairs from ChEMBL and BindingDB with pIC50 labels. Its index carries far more data for
our exact enzymes than the per-target ChEMBL pull does -- about 61,000 rows across the
four scored CYP accessions -- so it is the larger auxiliary source, with the same
interface as :mod:`cypshift.data.chembl` (``load_auxiliary`` returning SMILES, value,
target).

Two cautions, both measured (2026-09-06):

*deduplicate against the challenge*
    246 of the challenge's compounds appear in SAIR against some target, so a SAIR row
    matching a challenge compound is dropped -- otherwise the similarity-matched holdout
    leaks. Only 4 of the 750 blinded test compounds are in SAIR, so the competition
    itself is essentially clean, but the local split is not.
*already on the pIC50 scale*
    SAIR ships a ``pIC50`` column, so no unit conversion; rows outside a plausible range
    are dropped rather than clipped, as in the ChEMBL loader.
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cypshift.data.chembl import PIC50_RANGE, deduplicate_against
from cypshift.data.sequences import ACCESSIONS

logger = logging.getLogger(__name__)

#: The quantities :func:`load_auxiliary` can return (see :mod:`cypshift.data.sources`).
QUANTITIES = ("pic50",)

DEFAULT_INDEX = Path("data/external/sair.parquet")
DEFAULT_CACHE = Path("data/external")


def clean_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """SAIR rows for one target reduced to ``(SMILES, value)`` on the pIC50 scale.

    Replicates aggregate by median, as ChEMBL does, since SAIR keeps several cofolded
    structures per pair and several assays per compound.
    """
    if rows.empty:
        return pd.DataFrame(columns=["SMILES", "value"])
    frame = rows[["SMILES", "pIC50"]].copy()
    frame["pIC50"] = pd.to_numeric(frame["pIC50"], errors="coerce")
    frame = frame[frame["SMILES"].notna() & frame["pIC50"].notna()]
    low, high = PIC50_RANGE
    frame = frame[(frame["pIC50"] >= low) & (frame["pIC50"] <= high)]
    grouped = frame.groupby("SMILES")["pIC50"].median().reset_index()
    return grouped.rename(columns={"pIC50": "value"})


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    dedup_mode: str = "canonical_smiles",
    cache: Path = DEFAULT_CACHE,
    index: Path = DEFAULT_INDEX,
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated SAIR pIC50 rows for one enzyme.

    Cached as CSV per enzyme, since filtering the 8.8M-row index is the slow part and
    the result does not change between runs.

    Raises:
        KeyError: If the enzyme has no pinned UniProt accession.
        FileNotFoundError: If the SAIR index is absent (download it first).
    """
    if enzyme not in ACCESSIONS:
        raise KeyError(f"{enzyme} has no pinned UniProt accession")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"sair_{enzyme}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        if not index.exists():
            raise FileNotFoundError(f"SAIR index not found at {index}; download it first")
        accession = ACCESSIONS[enzyme]
        # Predicate pushdown: read only rows for this accession, not all 8.8M.
        rows = pq.read_table(
            index,
            columns=["SMILES", "pIC50", "protein"],
            filters=[("protein", "=", accession)],
        ).to_pandas()
        frame = clean_rows(rows)
        frame.to_csv(path, index=False)
        logger.info("cached %d SAIR compounds for %s (%s)", len(frame), enzyme, accession)
    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    return deduplicated
