"""Auxiliary CYP potencies from the Veith qHTS panel (PubChem, NCGC 2009).

Veith et al. (2009, https://www.nature.com/articles/nbt.1581) screened ~17k compounds
against five CYP isozymes in a bioluminescent (Promega P450-Glo) qHTS, depositing a full
concentration-response per compound. The per-isozyme deposits are:

    CYP1A2  -> AID 410     CYP2C9  -> AID 883     CYP2D6  -> AID 891
    CYP3A4  -> AID 884     CYP2C19 -> AID 899 (a non-scored relative)

Two honest caveats, both reasons to keep this a swept toggle rather than a default:

*not independent of ChEMBL.* ChEMBL's CYP ``AC50`` block is this same campaign, imported.
Our ChEMBL loader keeps only ``IC50``/``Ki`` (:mod:`cypshift.data.chembl`), so it does not
actually hold these rows -- but the *compounds* overlap, which the cross-source median
pooling in the pipeline collapses by canonical SMILES when both are enabled.

*a luminescent ranking signal, not the challenge assay.* The readout is pro-luciferin
dealkylation, not the challenge's fluorogenic probe; the assay mismatch is
structure-correlated, so treat Veith as a ranking prior, not a like-for-like potency.

Unlike ChEMBL/SAIR, the deposit ships the raw per-concentration points as well as the
fitted AC50. This loader currently emits the fitted potency as a pIC50 row (parity with
the other auxiliary sources); the raw points are a curve-model-specific extension left
for later.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.data.chembl import PIC50_RANGE, deduplicate_against
from cypshift.data.pubchem import download_assay

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/external/veith")

#: PubChem assay id per isozyme. Only these enzymes are covered; any other returns empty.
AID = {"CYP1A2": 410, "CYP2C9": 883, "CYP2D6": 891, "CYP3A4": 884, "CYP2C19": 899}

#: Curve fits we trust. A single active point or a poor fit is not a potency measurement,
#: so it is dropped rather than fed as one -- the same discipline ChEMBL censoring gets.
_BAD_CURVE = ("single point", "poor fit", "inactive")


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Veith records into ``(SMILES, value)`` pairs on the pIC50 scale.

    Two deposit schemas appear across the panel: the newer ones carry ``Potency`` (AC50
    in micromolar, so ``pIC50 = 6 - log10(Potency)``) and a ``Curve_Description``; the
    older CYP1A2 deposit carries ``Log of AC50`` (log10 molar, ``pIC50 = -value``) with an
    ``Activity Qualifier`` marking real (``=``) versus censored fits. Both keep only
    active compounds with a usable curve -- inactives and poor fits are dropped rather
    than turned into an invented potency at the detection limit.
    """
    frame = raw[raw["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"].copy()
    frame = frame[frame["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()]

    if "Potency" in frame.columns:
        potency = pd.to_numeric(frame["Potency"], errors="coerce")
        description = frame.get("Curve_Description", pd.Series("", index=frame.index)).fillna("")
        bad = description.str.lower().str.contains("|".join(_BAD_CURVE))
        frame = frame[(potency > 0) & ~bad]
        frame["value"] = 6.0 - np.log10(pd.to_numeric(frame["Potency"]).to_numpy(float))
    elif "Log of AC50" in frame.columns:
        qualifier = frame.get("Activity Qualifier", pd.Series("=", index=frame.index))
        log_ac50 = pd.to_numeric(frame["Log of AC50"], errors="coerce")
        frame = frame[(qualifier == "=") & log_ac50.notna()]
        frame["value"] = -pd.to_numeric(frame["Log of AC50"]).to_numpy(float)
    else:  # pragma: no cover -- guards against an unrecognised deposit schema
        raise KeyError("Veith deposit has neither 'Potency' nor 'Log of AC50'")

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
    enzyme: str, challenge_smiles: pd.Series, cache: Path = DEFAULT_CACHE
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Veith pIC50 rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``target``. Empty for any enzyme Veith does
    not cover, so the source contributes only where it has data. Downloads and caches on
    first use, since the PubChem paging is slow and the answer does not change.
    """
    if enzyme not in AID:
        return pd.DataFrame(columns=["SMILES", "value", "target"])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"veith_{enzyme}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = _clean(download_assay(AID[enzyme]))
        frame.to_csv(path, index=False)
        logger.info("Veith: cached %d %s compounds", len(frame), enzyme)
    deduplicated = deduplicate_against(frame, challenge_smiles)
    deduplicated["target"] = enzyme
    logger.info("Veith: %d %s compounds after dedup", len(deduplicated), enzyme)
    return deduplicated
