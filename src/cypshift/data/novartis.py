"""Novartis CYP3A4 time-dependent-inhibition data from the Polaris hub.

The one public source measured on the classification track's own endpoint: mechanism-based
(time-dependent) inactivation of CYP3A4, as ``log_kobs`` -- the log10 inactivation rate
constant from a preincubation time course, not a reversible pIC50. Roughly 16.7k rows,
CYP3A4 only (nothing for the 2D6 TDI track), CC-BY-4.0. Dataset ``novartis/novartis-cyp3a4-v1``
on https://polarishub.io ; methodology Zimmerlin et al. 2011
(https://pubmed.ncbi.nlm.nih.gov/21383203/), ML-scale version
https://pubs.acs.org/doi/10.1021/acs.chemrestox.3c00305.

Two prerequisites keep this out of a default run: the Polaris SDK must be installed
(``pip install polaris-lib``) and a Polaris Hub login is needed for the pull
(``polaris login``). The loader degrades gracefully -- a missing SDK or cache raises a
clear error rather than failing a run silently.

The label is a rate, not the reversible potency the regression auxiliary sources carry, so
this does not enter the pIC50 pool. Its natural use is the classification track: an
external CYP3A4 TDI set to validate the predicted ``is_TDI`` flag against, and -- once a
trainable TDI head exists -- to pretrain it.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.data.chembl import deduplicate_against

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/external/novartis/cyp3a4_tdi.csv")
NOVARTIS_ENZYME = "CYP3A4"
DATASET_ID = "novartis/novartis-cyp3a4-v1"

#: log10(k_obs) above which a compound is called a time-dependent inactivator. A
#: placeholder pending calibration against the challenge's own shift-based definition;
#: exposed so it can be swept rather than hard-coded into a label.
DEFAULT_LOG_KOBS_THRESHOLD = -2.0


def _download() -> pd.DataFrame:
    """Pull the Novartis CYP3A4 dataset from Polaris into ``(SMILES, log_kobs)``.

    Raises:
        ImportError: If the Polaris SDK is not installed.
    """
    try:
        import polaris as po
    except ImportError as error:  # pragma: no cover -- exercised only without the SDK
        raise ImportError(
            "the Polaris SDK is required for the Novartis TDI set; "
            "pip install polaris-lib and run 'polaris login'"
        ) from error

    dataset = po.load_dataset(DATASET_ID)
    frame = dataset.table if hasattr(dataset, "table") else pd.DataFrame(dataset)
    smiles_column = next(c for c in frame.columns if c.lower() in ("smiles", "canonical_smiles"))
    kobs_column = next(c for c in frame.columns if "kobs" in c.lower())
    return pd.DataFrame(
        {"SMILES": frame[smiles_column], "log_kobs": pd.to_numeric(frame[kobs_column], "coerce")}
    ).dropna()


def load(
    challenge_smiles: pd.Series,
    cache: Path = DEFAULT_CACHE,
    threshold: float = DEFAULT_LOG_KOBS_THRESHOLD,
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Novartis CYP3A4 TDI rows.

    Returns columns ``SMILES``, ``log_kobs``, ``is_TDI`` and ``target``. Downloads and
    caches on first use; a missing SDK or cache raises rather than failing a run silently.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        frame = pd.read_csv(cache)
    else:
        frame = _download()
        frame.to_csv(cache, index=False)
        logger.info("Novartis: cached %d CYP3A4 TDI rows", len(frame))
    frame = frame[np.isfinite(frame["log_kobs"])].copy()
    frame["is_TDI"] = frame["log_kobs"] > threshold
    deduplicated = deduplicate_against(frame, challenge_smiles)
    deduplicated["target"] = NOVARTIS_ENZYME
    logger.info("Novartis: %d CYP3A4 TDI rows after dedup", len(deduplicated))
    return deduplicated
