"""Auxiliary CYP potencies from the 2025 NCATS Tox21 P450-Glo re-screen (PubChem).

NCATS re-screened the Tox21 10K library against five CYP isozymes in a bioluminescent
P450-Glo qHTS and deposited the per-replicate concentration-response in 2025. Two reasons
this is worth adding where the older Tox21 CYP data was not:

*it is biochemical, not a cell reporter.* Despite the deposit titles ("luciferase reporter
qHTS"), the assay is a cell-free recombinant-enzyme inhibition measurement, so it does not
carry the cell-based luciferase-reporter artifact the earlier Tox21 CYP assays did; the
campaign also ran an explicit luciferase counterscreen.

*it is a distinct library from Veith.* The Tox21 10K collection is environmental-chemical
and drug heavy, not the MLSMR set behind Veith 2009, so this is genuinely new on-target
IC50 data rather than the Veith panel re-hosted. Overlap with the challenge and with
ChEMBL still exists and is deduplicated by canonical SMILES.

Per-isozyme deposits (the sixth AID in the range, 1671200, is an hERG assay and excluded):

    CYP1A2  -> AID 1671199    CYP2C9  -> AID 1671198    CYP2C19 -> AID 1671197
    CYP2D6  -> AID 1671196    CYP3A4  -> AID 1671201

Each compound is measured in many replicates; the deposit stores a ``Potency`` (AC50 in
micromolar) per replicate with no consolidated column, so the loader takes the per-compound
median. Ref: Huang et al. 2025, https://www.mdpi.com/1422-0067/26/11/4976.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.data.chembl import PIC50_RANGE, deduplicate_against
from cypshift.data.pubchem import download_assay

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/external/tox21")

#: PubChem assay id per isozyme. Only these enzymes are covered; any other returns empty.
AID = {
    "CYP1A2": 1671199,
    "CYP2C9": 1671198,
    "CYP2C19": 1671197,
    "CYP2D6": 1671196,
    "CYP3A4": 1671201,
}


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Tox21 re-screen records into ``(SMILES, value)`` pairs on the pIC50 scale.

    Keeps active compounds and takes the median ``Potency`` (AC50 in micromolar) across
    the replicate columns, so ``pIC50 = 6 - log10(median potency)``. Inactives -- which
    carry no potency -- are dropped rather than turned into an invented detection-limit
    value.
    """
    frame = raw[raw["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"].copy()
    frame = frame[frame["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()]
    potency_columns = [c for c in frame.columns if c.startswith("Potency-Replicate")]
    if not potency_columns:
        raise KeyError("Tox21 deposit has no 'Potency-Replicate' columns")
    potency = frame[potency_columns].apply(pd.to_numeric, errors="coerce")
    median = potency.where(potency > 0).median(axis=1)

    frame = frame[median.notna()]
    frame["value"] = 6.0 - np.log10(median[median.notna()].to_numpy(float))
    low, high = PIC50_RANGE
    frame = frame[(frame["value"] >= low) & (frame["value"] <= high)]
    grouped = (
        frame.groupby("PUBCHEM_EXT_DATASOURCE_SMILES")["value"]
        .median()
        .reset_index()
        .rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
    )
    return grouped


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    cache: Path = DEFAULT_CACHE,
    dedup_mode: str = "canonical_smiles",
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Tox21 pIC50 rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``target``. Empty for any enzyme Tox21 does not
    cover. Downloads and caches on first use, since the PubChem paging is slow and the
    answer does not change.
    """
    if enzyme not in AID:
        return pd.DataFrame(columns=["SMILES", "value", "target"])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"tox21_{enzyme}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = _clean(download_assay(AID[enzyme]))
        frame.to_csv(path, index=False)
        logger.info("Tox21: cached %d %s compounds", len(frame), enzyme)
    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    logger.info("Tox21: %d %s compounds after dedup", len(deduplicated), enzyme)
    return deduplicated
