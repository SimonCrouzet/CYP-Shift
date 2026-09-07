"""Bulk download of a PubChem BioAssay's records, paged by substance id.

The direct ``.../aid/<id>/CSV`` endpoint rejects large assays with "Too many SIDs", so the
substance list is fetched first and the data pulled in POST batches. Shared by the PubChem
qHTS auxiliary sources (:mod:`cypshift.data.veith`, :mod:`cypshift.data.tox21`).
"""

import io
import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PUBCHEM_ASSAY = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid"


def download_assay(
    aid: int, chunk: int = 4000, timeout: int = 300, pause: float = 0.2
) -> pd.DataFrame:
    """Every record for one PubChem assay, concatenated across SID batches."""
    sid_response = requests.get(f"{PUBCHEM_ASSAY}/{aid}/sids/JSON", timeout=timeout)
    sid_response.raise_for_status()
    sids = sid_response.json()["InformationList"]["Information"][0]["SID"]
    frames = []
    for start in range(0, len(sids), chunk):
        batch = sids[start : start + chunk]
        response = requests.post(
            f"{PUBCHEM_ASSAY}/{aid}/CSV",
            data={"sid": ",".join(map(str, batch))},
            timeout=timeout,
        )
        response.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(response.text), low_memory=False))
        logger.info("  AID %d: %d of %d SIDs", aid, min(start + chunk, len(sids)), len(sids))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)
