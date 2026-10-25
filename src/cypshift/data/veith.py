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
fitted AC50. Actives enter as a fitted pIC50 row; with ``include_inactives`` the panel's
inactive compounds are kept as censored rows below the top screened concentration
(read from the deposit), which corrects the active-only selection bias. Feeding the raw
points to the curve model is a further extension left for later.
"""

import logging
import re
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

#: The per-concentration readout columns the deposit ships, e.g. ``Activity at 57.14 uM``.
#: The highest of these is the top screened concentration, read from the data rather than
#: hard-coded, so the inactive detection limit follows whatever a given deposit used.
_CONCENTRATION_COLUMN = re.compile(r"Activity at ([0-9.]+)\s*uM")


def detection_limit_pic50(raw: pd.DataFrame) -> float | None:
    """The pIC50 below which an inactive compound's AC50 must lie, from the deposit.

    An inactive did not reach its AC50 at or below the top screened concentration, so its
    pIC50 is below ``6 - log10(top_uM)``. The top concentration is the largest value named
    in the ``Activity at <conc> uM`` columns. Returns ``None`` for a deposit that ships no
    such columns, since the limit cannot then be read from the data.
    """
    concentrations = [
        float(match.group(1))
        for column in raw.columns
        if (match := _CONCENTRATION_COLUMN.fullmatch(str(column)))
    ]
    return None if not concentrations else 6.0 - float(np.log10(max(concentrations)))


def _max_response(named: pd.DataFrame) -> pd.DataFrame:
    """All screened compounds as a 0-1 inhibition magnitude at the maximal curve response.

    ``Max_Response`` is a signed percent activity, negative for inhibition (AID 883: Active
    median -82.4, Inactive median +0.6), so the value is ``-Max_Response / 100`` clipped to
    [0, 1]; the activation side clips to 0. The inactives stay as real low-response
    measurements. This is the signal the graph model's auxiliary response head is trained on,
    a different quantity from potency, so it never shares the pIC50 scale.
    """
    if "Max_Response" not in named.columns:
        raise KeyError("cannot use max_response: deposit has no 'Max_Response' column")
    screened = named[named["PUBCHEM_ACTIVITY_OUTCOME"].isin(["Active", "Inactive"])].copy()
    screened["value"] = (-pd.to_numeric(screened["Max_Response"], errors="coerce") / 100.0).clip(
        0, 1
    )
    screened = screened.dropna(subset=["value"])
    grouped = (
        screened.groupby("PUBCHEM_EXT_DATASOURCE_SMILES")["value"]
        .median()
        .reset_index()
        .rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
    )
    grouped["conf_low"] = grouped["value"]
    grouped["conf_high"] = grouped["value"]
    grouped["is_censored"] = False
    return grouped


def _clean(
    raw: pd.DataFrame, include_inactives: bool = False, representation: str = "pic50"
) -> pd.DataFrame:
    """Veith records into ``SMILES, value, conf_low, conf_high, is_censored`` on the pIC50 scale.

    Two deposit schemas appear across the panel: the newer ones carry ``Potency`` (AC50
    in micromolar, so ``pIC50 = 6 - log10(Potency)``) and a ``Curve_Description``; the
    older CYP1A2 deposit carries ``Log of AC50`` (log10 molar, ``pIC50 = -value``) with an
    ``Activity Qualifier`` marking real (``=``) versus censored fits. Active compounds with
    a usable curve become point values.

    With ``include_inactives``, the panel's inactive compounds are kept as censored rows:
    their AC50 lies above the top screened concentration, so ``pIC50 < LIMIT_PIC50`` is a
    real one-sided observation (``conf_high`` at the limit, ``is_censored`` true), not an
    invented potency. Keeping them corrects the active-only selection bias; dropping them
    (the default-off path) reproduces the earlier behaviour.
    """
    named = raw[raw["PUBCHEM_EXT_DATASOURCE_SMILES"].notna()]
    if representation == "max_response":
        return _max_response(named)
    frame = named[named["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"].copy()

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
    active = (
        frame.groupby("PUBCHEM_EXT_DATASOURCE_SMILES")["value"]
        .median()
        .reset_index()
        .rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "SMILES"})
    )
    active["conf_low"] = active["value"]
    active["conf_high"] = active["value"]
    active["is_censored"] = False
    if not include_inactives:
        return active

    limit = detection_limit_pic50(raw)
    if limit is None:
        raise KeyError("cannot keep inactives: deposit has no 'Activity at <conc> uM' columns")
    inactive_smiles = (
        named.loc[named["PUBCHEM_ACTIVITY_OUTCOME"] == "Inactive", "PUBCHEM_EXT_DATASOURCE_SMILES"]
        .dropna()
        .unique()
    )
    fresh = [s for s in inactive_smiles if s not in set(active["SMILES"])]
    inactive = pd.DataFrame({"SMILES": fresh})
    inactive["value"] = limit
    inactive["conf_low"] = limit
    inactive["conf_high"] = limit
    inactive["is_censored"] = True
    return pd.concat([active, inactive], ignore_index=True)


def load_auxiliary(
    enzyme: str,
    challenge_smiles: pd.Series,
    dedup_mode: str = "canonical_smiles",
    cache: Path = DEFAULT_CACHE,
    include_inactives: bool = False,
    representation: str = "pic50",
) -> pd.DataFrame:
    """Cleaned, challenge-deduplicated Veith rows for one enzyme.

    Returns columns ``SMILES``, ``value``, ``conf_low``, ``conf_high``, ``is_censored``,
    ``target``. ``representation`` chooses the quantity: ``pic50`` (fitted potency, with
    ``include_inactives`` adding censored rows) or ``max_response`` (a 0-1 response for the
    graph's auxiliary head). Empty for any enzyme Veith does not cover. Downloads and caches
    on first use; each representation caches separately.
    """
    columns = ["SMILES", "value", "conf_low", "conf_high", "is_censored", "target"]
    if enzyme not in AID:
        return pd.DataFrame(columns=columns)
    cache.mkdir(parents=True, exist_ok=True)
    # "_response_inhibition" marks the sign-corrected response: the "_response" caches written
    # before it hold ``+Max_Response / 100`` (inhibitors at 0) and are not reused.
    suffix = (
        "_response_inhibition"
        if representation == "max_response"
        else ("_full" if include_inactives else "")
    )
    path = cache / f"veith_{enzyme}{suffix}.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = _clean(
            download_assay(AID[enzyme]),
            include_inactives=include_inactives,
            representation=representation,
        )
        frame.to_csv(path, index=False)
        logger.info(
            "Veith: cached %d %s compounds (inactives=%s)", len(frame), enzyme, include_inactives
        )
    deduplicated = deduplicate_against(frame, challenge_smiles, dedup_mode)
    deduplicated["target"] = enzyme
    logger.info("Veith: %d %s compounds after dedup", len(deduplicated), enzyme)
    return deduplicated
