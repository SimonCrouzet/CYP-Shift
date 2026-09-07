"""Tests for the Veith qHTS auxiliary loader (no network: caches are pre-written)."""

import pandas as pd
import pytest

from cypshift.data.veith import AID, _clean, load_auxiliary


def test_potency_schema_maps_to_pic50():
    """Newer deposits: pIC50 = 6 - log10(Potency[uM]); poor fits and inactives dropped."""
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Active", "Inactive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCN", "CCC"],
            "Potency": [1.0, 10.0, 0.5],
            "Curve_Description": ["Complete curve; high efficacy", "Single point of activity", ""],
        }
    )
    cleaned = _clean(raw)
    assert set(cleaned["SMILES"]) == {"CCO"}  # CCN is a single point, CCC is inactive
    assert cleaned.set_index("SMILES").loc["CCO", "value"] == pytest.approx(6.0)


def test_logac50_schema_maps_to_pic50():
    """Older CYP1A2 deposit: pIC50 = -(Log of AC50), keep only the '=' qualifier."""
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Active"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCN"],
            "Log of AC50": [-6.2, -5.0],
            "Activity Qualifier": ["=", ">"],
        }
    )
    cleaned = _clean(raw)
    assert set(cleaned["SMILES"]) == {"CCO"}  # CCN is censored ('>')
    assert cleaned.set_index("SMILES").loc["CCO", "value"] == pytest.approx(6.2)


def test_unrecognised_schema_raises():
    raw = pd.DataFrame(
        {"PUBCHEM_ACTIVITY_OUTCOME": ["Active"], "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"]}
    )
    with pytest.raises(KeyError, match="Potency"):
        _clean(raw)


def test_uncovered_enzyme_is_empty(tmp_path):
    assert load_auxiliary("CYP2E1", pd.Series(["CCO"]), cache=tmp_path).empty


def test_reads_cache_and_deduplicates(tmp_path):
    (tmp_path / "veith_CYP2C9.csv").write_text("SMILES,value\nCCO,6.0\nCCN,7.0\n")
    kept = load_auxiliary("CYP2C9", pd.Series(["OCC"]), cache=tmp_path)  # OCC == CCO canonical
    assert set(kept["SMILES"]) == {"CCN"}
    assert (kept["target"] == "CYP2C9").all()


def test_aid_map_covers_the_four_scored_enzymes():
    assert {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"} <= set(AID)
