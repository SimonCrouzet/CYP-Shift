"""Fetching auxiliary enzyme activity from ChEMBL.

The auxiliary machinery in :mod:`cypshift.data.auxiliary` selects enzymes by
binding-site distance and weights their rows by it, and until now had nothing to
weight: the challenge files cover only the four scored enzymes. This is the source for
the rest.

Only CYP2C19 is clearly worth it. Its binding site sits 0.53 from CYP2C9's -- closer
than a single conservative Leu-to-Ile substitution per position -- and it has about
6,000 IC50/Ki records here plus full coverage in the Veith panel. The next candidates
fall away fast: CYP2C8 at 1.28 with 1,151 records, CYP1A1 at 0.87 with 788, and
CYP3A4's three closest relatives have five, 134 and two records between them.

Three things are done to the raw records, each of which changes the answer:

*deduplicate against the challenge set*
    A compound measured in both would otherwise appear as an auxiliary row and a
    scored label, and a model would be rewarded for memorising it. Matching is on
    canonical SMILES, since the same compound reaches the two sources under different
    names.
*keep only real potencies*
    ChEMBL carries censored records (``>``, ``<``) and non-nanomolar units. A ``>10000
    nM`` reading is a statement that the compound is inactive, not a measurement of
    10 uM, and treating it as one invents potency that was never observed.
*aggregate replicates*
    The median across records for a compound and target, since ChEMBL aggregates many
    papers and the mean is pulled by a single outlying assay.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cypshift.data.sequences import ACCESSIONS

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_CACHE = Path("data/external")

#: Only these are accepted as a potency. ChEMBL mixes assay types under one target.
POTENCY_TYPES = ("IC50", "Ki")

#: Records outside this are dropped rather than clipped. Below it the value is usually
#: a typo or a different unit; above it the compound is inactive and the number is a
#: detection limit, not a measurement.
PIC50_RANGE = (3.0, 11.0)


def target_id(accession: str, timeout: int = 60) -> str | None:
    """The single-protein ChEMBL target for a UniProt accession."""
    response = requests.get(
        f"{CHEMBL_API}/target.json",
        params={"target_components__accession": accession, "limit": 20},
        timeout=timeout,
    )
    response.raise_for_status()
    single = [
        t["target_chembl_id"]
        for t in response.json().get("targets", [])
        if t.get("target_type") == "SINGLE PROTEIN"
    ]
    if not single:
        logger.info("no single-protein ChEMBL target for %s", accession)
        return None
    return single[0]


def fetch_activities(
    chembl_id: str, page_size: int = 1000, timeout: int = 120, pause: float = 0.2
) -> pd.DataFrame:
    """Every IC50 and Ki record for a target, paged through the API."""
    rows: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            f"{CHEMBL_API}/activity.json",
            params={
                "target_chembl_id": chembl_id,
                "standard_type__in": ",".join(POTENCY_TYPES),
                "limit": page_size,
                "offset": offset,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("activities", [])
        rows.extend(batch)
        total = payload["page_meta"]["total_count"]
        offset += page_size
        logger.info("  %s: %d of %d records", chembl_id, min(offset, total), total)
        if offset >= total or not batch:
            break
        time.sleep(pause)
    return pd.DataFrame(rows)


def to_pic50(activities: pd.DataFrame) -> pd.DataFrame:
    """Clean records into ``(SMILES, value)`` pairs on the pIC50 scale.

    Censored records and non-nanomolar units are dropped rather than converted. A
    ``>10000 nM`` reading says the compound is inactive; turning it into pIC50 5.0
    would assert a potency nobody measured.
    """
    if activities.empty:
        return pd.DataFrame(columns=["SMILES", "value"])
    frame = activities.copy()
    before = len(frame)
    frame = frame[frame["standard_relation"].isin(["=", "=="])]
    frame = frame[frame["standard_units"] == "nM"]
    frame = frame[frame["canonical_smiles"].notna() & frame["standard_value"].notna()]
    frame["standard_value"] = pd.to_numeric(frame["standard_value"], errors="coerce")
    frame = frame[frame["standard_value"] > 0]

    frame["value"] = 9.0 - np.log10(frame["standard_value"].to_numpy(float))
    low, high = PIC50_RANGE
    frame = frame[(frame["value"] >= low) & (frame["value"] <= high)]
    logger.info("kept %d of %d records after cleaning", len(frame), before)

    grouped = (
        frame.groupby("canonical_smiles")["value"]
        .median()
        .reset_index()
        .rename(columns={"canonical_smiles": "SMILES"})
    )
    return grouped


def canonical(smiles: pd.Series) -> pd.Series:
    """RDKit canonical SMILES, for matching across sources that differ in spelling."""
    from rdkit import Chem

    def one(text: str) -> str | None:
        molecule = Chem.MolFromSmiles(text)
        return Chem.MolToSmiles(molecule) if molecule else None

    return smiles.map(one)


def dedup_key(smiles: pd.Series, mode: str = "canonical_smiles") -> pd.Series:
    """The identity a compound is matched by against the challenge set.

    ``canonical_smiles`` matches exact structures; ``inchikey_block`` matches on the InChIKey's
    first (connectivity) block, so salts, stereoisomers and charge/isotope variants of one
    skeleton collapse together -- a coarser, more aggressive dedup.
    """
    if mode == "canonical_smiles":
        return canonical(smiles)
    from rdkit import Chem

    def one(text: str) -> str | None:
        molecule = Chem.MolFromSmiles(text)
        return Chem.MolToInchiKey(molecule).split("-")[0] if molecule else None

    return smiles.map(one)


def deduplicate_against(
    frame: pd.DataFrame, challenge_smiles: pd.Series, mode: str = "canonical_smiles"
) -> pd.DataFrame:
    """Drop auxiliary compounds that also appear in the challenge set.

    Without this a compound measured in both is an auxiliary training row and a scored
    label, and the model is rewarded for memorising it rather than for generalising. ``mode``
    picks the matching identity (see :func:`dedup_key`).
    """
    known = set(dedup_key(challenge_smiles, mode).dropna())
    frame = frame.copy()
    frame["_dedup_key"] = dedup_key(frame["SMILES"], mode)
    overlap = frame["_dedup_key"].isin(known)
    logger.info(
        "dropping %d of %d compounds also in the challenge set", int(overlap.sum()), len(frame)
    )
    return frame.loc[~overlap & frame["_dedup_key"].notna()].drop(columns="_dedup_key")


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    cache: Path = DEFAULT_CACHE,
    dedup_mode: str = "canonical_smiles",
) -> pd.DataFrame:
    """Cleaned, deduplicated pIC50 records for one auxiliary enzyme.

    Cached as CSV, since the API paging is slow and the answer does not change between
    runs.

    Raises:
        KeyError: If the enzyme has no pinned UniProt accession.
    """
    if enzyme not in ACCESSIONS:
        raise KeyError(f"{enzyme} has no pinned UniProt accession")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"chembl_{enzyme}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        chembl_id = target_id(ACCESSIONS[enzyme])
        if chembl_id is None:
            return pd.DataFrame(columns=["SMILES", "value"])
        frame = to_pic50(fetch_activities(chembl_id))
        frame.to_csv(path, index=False)
        logger.info("cached %d compounds for %s", len(frame), enzyme)
    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    return deduplicated
