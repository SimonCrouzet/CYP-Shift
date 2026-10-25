"""Auxiliary CYP3A4 data from the Octant / OpenADMET blog release.

This is the closest thing to more challenge data: the same Octant Supersome dose-response
platform the challenge itself uses, so the assay noise matches rather than being a
foreign mix like ChEMBL or SAIR. It covers **CYP3A4 only** (1,340 compounds, 1,139 QC
PASS), and -- unlike the other auxiliary sources -- it ships a full curve: pIC50 with a
credible interval, Hill slope and Emax. The interval is kept rather than collapsed to a
point, so the soft-threshold metric sees a real tolerance on these rows.

Measured overlap with the challenge (canonical SMILES): 14 of the training compounds and
**1 of the 750 blinded test compounds**, so it is essentially clean; the overlap is
deduplicated regardless.

Source: https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release
(CC-BY-4.0). Its 30-minute active-enzyme preincubation means the readout captures
reversible plus metabolite/time-dependent inhibition -- relevant to the CYP3A4 TDI assay_condition.
"""

import logging
from pathlib import Path

import pandas as pd

from cypshift.data.chembl import PIC50_RANGE, deduplicate_against

logger = logging.getLogger(__name__)

#: The quantities :func:`load_auxiliary` can return (see :mod:`cypshift.data.sources`).
QUANTITIES = ("pic50",)

DEFAULT_TSV = Path("data/external/octant/inhibition.tsv")
#: The only enzyme this source covers.
OCTANT_ENZYME = "CYP3A4"


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    dedup_mode: str = "canonical_smiles",
    cache: Path = DEFAULT_TSV,
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Octant rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``target``, ``conf_low``, ``conf_high`` --
    the credible interval preserved. Empty for any enzyme other than CYP3A4, so the
    source contributes only where it has data.

    Raises:
        FileNotFoundError: If the Octant TSV is absent (download it first).
    """
    if enzyme != OCTANT_ENZYME:
        return pd.DataFrame(columns=["SMILES", "value", "target", "conf_low", "conf_high"])
    if not cache.exists():
        raise FileNotFoundError(f"Octant data not found at {cache}; download it first")

    raw = pd.read_csv(cache, sep="\t")
    frame = raw[raw["drc_qc_status"] == "PASS"].copy()
    frame = frame.rename(
        columns={
            "standardized_smiles": "SMILES",
            "CYP3A4_pIC50": "value",
            "CYP3A4_pIC50_ci_lower": "conf_low",
            "CYP3A4_pIC50_ci_upper": "conf_high",
        }
    )
    frame = frame[frame["SMILES"].notna() & frame["value"].notna()]
    low, high = PIC50_RANGE
    frame = frame[(frame["value"] >= low) & (frame["value"] <= high)]
    frame = frame[["SMILES", "value", "conf_low", "conf_high"]]
    # A row without a stored interval falls back to a point rather than an invented span.
    frame["conf_low"] = frame["conf_low"].fillna(frame["value"])
    frame["conf_high"] = frame["conf_high"].fillna(frame["value"])

    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    logger.info("Octant: %d CYP3A4 compounds after cleaning and dedup", len(deduplicated))
    return deduplicated
