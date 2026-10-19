"""Novartis CYP inhibition data from the Polaris hub.

One dataset, ``novartis/novartis-cyp3a4-v1`` on https://polarishub.io, that turns out to
carry two kinds of signal over ~16.7k compounds (CC-BY-4.0, and it loads anonymously --
no Polaris login needed for a public dataset):

*reversible potency* -- ``pIC50_CYP3A4``, ``pIC50_CYP2C9`` and ``pIC50_CYP2D6``, real
reversible inhibition on three of the four scored enzymes. This is the same quantity the
regression track predicts, so it enters the auxiliary pool like the other potency sources
-- and it is a large, independent lever for CYP2D6, the scored enzyme with no near relative.

*time-dependent inhibition* -- ``log_kobs`` (the log10 inactivation rate constant from a
preincubation time course) and ``CLS_log_kobs`` (a 0/1/2 severity class), on CYP3A4 only.
That is the classification track's own endpoint, a rate rather than a pIC50, exposed by
:func:`load_tdi` for that track rather than mixed into the potency pool.

Methodology Zimmerlin et al. 2011 (https://pubmed.ncbi.nlm.nih.gov/21383203/); ML-scale
version https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00305.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.data.chembl import PIC50_RANGE, deduplicate_against

logger = logging.getLogger(__name__)

DATASET_ID = "novartis/novartis-cyp3a4-v1"
DEFAULT_CACHE = Path("data/external/novartis/cyp3a4_v1.csv")
SMILES_COLUMN = "MOL_smiles"

#: The scored enzymes this dataset carries a reversible pIC50 column for.
NOVARTIS_ENZYMES = ("CYP3A4", "CYP2C9", "CYP2D6")


def _table(cache: Path) -> pd.DataFrame:
    """The Novartis table, downloaded via Polaris on first use and cached as parquet.

    Raises:
        ImportError: If the Polaris SDK is not installed.
    """
    if cache.exists():
        return pd.read_csv(cache)
    try:
        import polaris as po
    except ImportError as error:  # pragma: no cover -- exercised only without the SDK
        raise ImportError(
            "the Polaris SDK is required to fetch the Novartis dataset; pip install polaris-lib"
        ) from error
    dataset = po.load_dataset(DATASET_ID)
    frame = dataset.table if hasattr(dataset, "table") else pd.DataFrame(dataset)
    keep = [SMILES_COLUMN, "log_kobs", "CLS_log_kobs", *(f"pIC50_{e}" for e in NOVARTIS_ENZYMES)]
    frame = frame[[c for c in keep if c in frame.columns]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    logger.info("Novartis: cached %d rows", len(frame))
    return frame


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    cache: Path = DEFAULT_CACHE,
    dedup_mode: str = "canonical_smiles",
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated reversible pIC50 rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``target``. Empty for any enzyme the dataset
    does not carry a pIC50 column for (only CYP3A4/2C9/2D6 are covered), so the source
    contributes only where it has data.
    """
    if enzyme not in NOVARTIS_ENZYMES:
        return pd.DataFrame(columns=["SMILES", "value", "target"])
    frame = _table(cache)
    value = pd.to_numeric(frame[f"pIC50_{enzyme}"], errors="coerce")
    clean = pd.DataFrame({"SMILES": frame[SMILES_COLUMN], "value": value})
    clean = clean[clean["SMILES"].notna() & np.isfinite(clean["value"])]
    low, high = PIC50_RANGE
    clean = clean[(clean["value"] >= low) & (clean["value"] <= high)]
    clean = clean.groupby("SMILES", as_index=False)["value"].median()
    deduplicated = deduplicate_against(clean, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    logger.info("Novartis: %d %s pIC50 rows after dedup", len(deduplicated), enzyme)
    return deduplicated


def load_tdi(
    challenge_smiles: pd.Series, cache: Path = DEFAULT_CACHE, dedup_mode: str = "canonical_smiles"
) -> pd.DataFrame:
    """Challenge-deduplicated CYP3A4 time-dependent-inhibition rows.

    Returns ``SMILES``, ``log_kobs`` (the log10 inactivation rate) and ``tdi_class`` (the
    deposited 0/1/2 severity). For the classification track, kept apart from the potency
    pool because the label is a rate, not a reversible pIC50.
    """
    frame = _table(cache)
    clean = pd.DataFrame(
        {
            "SMILES": frame[SMILES_COLUMN],
            "log_kobs": pd.to_numeric(frame["log_kobs"], errors="coerce"),
            "tdi_class": pd.to_numeric(frame["CLS_log_kobs"], errors="coerce"),
        }
    )
    clean = clean[clean["SMILES"].notna() & np.isfinite(clean["log_kobs"])]
    return deduplicate_against(clean, challenge_smiles, dedup_mode)
