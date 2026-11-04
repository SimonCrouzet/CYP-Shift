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


def max_response_inhibition(records: pd.DataFrame, response_columns: list[str]) -> pd.DataFrame:
    """Active and Inactive compounds as a 0-1 inhibition magnitude at the maximal curve response.

    ``Max_Response`` is a signed percent activity, negative for inhibition (Veith AID 883: Active
    median -82.4, Inactive +0.6; Tox21 AID 1671199: Active -78.0, Inactive -14.7). Per record the
    value is the median over ``response_columns`` (one column, or one per replicate) of
    ``-Max_Response / 100`` clipped to [0, 1], so the activation side clips to 0; per compound it
    is the median over its records. Inconclusive compounds are dropped.

    Returns ``SMILES, value, conf_low, conf_high, is_censored``.
    """
    smiles_column = "PUBCHEM_EXT_DATASOURCE_SMILES"
    screened = records[
        records[smiles_column].notna()
        & records["PUBCHEM_ACTIVITY_OUTCOME"].isin(["Active", "Inactive"])
    ].copy()
    response = screened[response_columns].apply(pd.to_numeric, errors="coerce").median(axis=1)
    screened["value"] = (-response / 100.0).clip(0, 1)
    grouped = (
        screened.dropna(subset=["value"])
        .groupby(smiles_column)["value"]
        .median()
        .reset_index()
        .rename(columns={smiles_column: "SMILES"})
    )
    grouped["conf_low"] = grouped["value"]
    grouped["conf_high"] = grouped["value"]
    grouped["is_censored"] = False
    return grouped
