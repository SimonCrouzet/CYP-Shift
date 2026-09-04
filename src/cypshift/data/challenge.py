"""Fetching and loading the OpenADMET CYP challenge datasets.

Five files are published at
https://huggingface.co/datasets/openadmet/cyp-challenge-train-test. They are not
interchangeable, and which one to reach for matters:

``inhibition``
    Dose-response pIC50 with 95% credible bounds. The supervised target, but coverage
    is 26-48% per isoform and is *not* missing at random: a compound was promoted to a
    full curve only if it hit that isoform in the primary screen.
``screen``
    Single-concentration primary screen at 49.5 uM, dense over 4,376 compounds by four
    enzymes. Unbiased, and its log2 fold-change ranks pIC50 at Spearman 0.83-0.94, so
    it supplies the inactives the dose-response labels lack.
``tdi``
    Time-dependent inhibition arm, with paired direct-inhibition values.
``emax``
    Maximal effect. Its 5th-95th percentile spans about 0.16 log units on every
    isoform, so it carries no discriminative signal; kept for completeness.
``test``
    The 750 blinded compounds, SMILES only.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

from cypshift.data.manifest import Manifest, verify

logger = logging.getLogger(__name__)

HF_REPO = "openadmet/cyp-challenge-train-test"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

FILENAMES = {
    "inhibition": "cyp-challenge-TRAIN_inhibition.csv",
    "screen": "cyp-challenge-single-concentration-TRAIN.csv",
    "tdi": "cyp-challenge-TRAIN_TDI.csv",
    "emax": "cyp-challenge-TRAIN_Emax.csv",
    "test": "cyp-challenge-TEST-BLINDED.csv",
}

DEFAULT_DATA_DIR = Path("data/raw")


def fetch(
    name: str,
    directory: Path = DEFAULT_DATA_DIR,
    force: bool = False,
    timeout_seconds: int = 300,
) -> Path:
    """Download one challenge file and record its provenance.

    Args:
        name: A key of :data:`FILENAMES`.
        directory: Where to write the file and its manifest.
        force: Re-download even if a verified copy is already present.
        timeout_seconds: Per-request timeout.

    Returns:
        Path to the downloaded file.
    """
    if name not in FILENAMES:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(FILENAMES)}")

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAMES[name]
    if path.exists() and not force and verify(path):
        logger.info("%s already present and verified", path.name)
        return path

    url = f"{HF_BASE_URL}/{FILENAMES[name]}"
    logger.info("downloading %s", url)
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    path.write_bytes(response.content)

    frame = pd.read_csv(path)
    Manifest.describe(path, url, n_rows=len(frame), n_columns=frame.shape[1]).write(directory)
    logger.info("wrote %s (%d rows, %d columns)", path.name, len(frame), frame.shape[1])
    return path


def fetch_all(directory: Path = DEFAULT_DATA_DIR, force: bool = False) -> dict[str, Path]:
    """Download every challenge file. Roughly 6.6 MB in total."""
    return {name: fetch(name, directory, force) for name in FILENAMES}


def load(name: str, directory: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load one challenge file, downloading it first if absent."""
    return pd.read_csv(fetch(name, directory))
