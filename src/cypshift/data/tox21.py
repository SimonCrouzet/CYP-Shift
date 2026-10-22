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

#: The censoring bound for a compound the campaign called inactive, in pIC50. It is the
#: assay's highest tested concentration, not the pIC50 plausibility filter: "no inhibition
#: resolved up to the top dose" means IC50 >= that dose, i.e. pIC50 <= this. The NTP protocol
#: table lists 115.1596 uM as the highest tested concentration for these five AIDs, so the
#: bound is 6 - log10(115.1596) ~= 3.94, not the 3.0 plausibility floor. See
#: https://www.ncbi.nlm.nih.gov/books/NBK592943/table/t-2-E.1/ .
INACTIVE_PIC50_BOUND = 6.0 - np.log10(115.1596)

#: PubChem assay id per isozyme. Only these enzymes are covered; any other returns empty.
AID = {
    "CYP1A2": 1671199,
    "CYP2C9": 1671198,
    "CYP2C19": 1671197,
    "CYP2D6": 1671196,
    "CYP3A4": 1671201,
}


def _clean(raw: pd.DataFrame, include_inactives: bool = False) -> pd.DataFrame:
    """Tox21 re-screen records into ``(SMILES, value, is_censored)`` rows on the pIC50 scale.

    Keeps active compounds and takes the median ``Potency`` (AC50 in micromolar) across the
    replicate columns, so ``pIC50 = 6 - log10(median potency)``. With ``include_inactives``,
    the compounds the campaign called inactive -- which carry no potency -- are kept as
    censored rows (``is_censored=True``) with value :data:`INACTIVE_PIC50_BOUND`, i.e. an upper
    bound on pIC50. Without the flag they are dropped (the historic behaviour), rather than
    turned into an invented point estimate.
    """
    smiles_column = "PUBCHEM_EXT_DATASOURCE_SMILES"
    with_smiles = raw[raw[smiles_column].notna()]
    active = with_smiles[with_smiles["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"].copy()
    potency_columns = [c for c in active.columns if c.startswith("Potency-Replicate")]
    if not potency_columns:
        raise KeyError("Tox21 deposit has no 'Potency-Replicate' columns")
    potency = active[potency_columns].apply(pd.to_numeric, errors="coerce")
    median = potency.where(potency > 0).median(axis=1)
    active = active[median.notna()]
    active["value"] = 6.0 - np.log10(median[median.notna()].to_numpy(float))
    low, high = PIC50_RANGE
    active = active[(active["value"] >= low) & (active["value"] <= high)]
    grouped = (
        active.groupby(smiles_column)["value"]
        .median()
        .reset_index()
        .rename(columns={smiles_column: "SMILES"})
    )
    grouped["is_censored"] = False
    if not include_inactives:
        return grouped

    # Inactives: one censored row per compound at the highest-tested-concentration bound,
    # excluding any compound that also has an active reading (the active measurement supersedes
    # the detection-limit bound).
    inactive_smiles = set(
        with_smiles.loc[with_smiles["PUBCHEM_ACTIVITY_OUTCOME"] == "Inactive", smiles_column]
    ) - set(grouped["SMILES"])
    if not inactive_smiles:
        return grouped
    censored = pd.DataFrame(
        {
            "SMILES": sorted(inactive_smiles),
            "value": float(INACTIVE_PIC50_BOUND),
            "is_censored": True,
        }
    )
    logger.info("Tox21: %d active, %d censored inactive compounds", len(grouped), len(censored))
    return pd.concat([grouped, censored], ignore_index=True)


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    cache: Path = DEFAULT_CACHE,
    dedup_mode: str = "canonical_smiles",
    include_inactives: bool = False,
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Tox21 pIC50 rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``is_censored``, ``target``. Empty for any enzyme
    Tox21 does not cover. Downloads and caches on first use, since the PubChem paging is slow
    and the answer does not change; the active-only and with-inactives cleanings are cached
    under separate names so one never masquerades as the other.
    """
    if enzyme not in AID:
        return pd.DataFrame(columns=["SMILES", "value", "is_censored", "target"])
    cache.mkdir(parents=True, exist_ok=True)
    # The suffix versions the cache: "_cmax" marks the highest-tested-concentration inactive
    # bound, so a cache written under the earlier plausibility-floor value is not reused.
    suffix = "_with_inactives_cmax" if include_inactives else ""
    path = cache / f"tox21_{enzyme}{suffix}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = _clean(download_assay(AID[enzyme]), include_inactives=include_inactives)
        frame.to_csv(path, index=False)
        logger.info("Tox21: cached %d %s compounds", len(frame), enzyme)
    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    logger.info("Tox21: %d %s compounds after dedup", len(deduplicated), enzyme)
    return deduplicated
