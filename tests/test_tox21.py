"""Tests for the Tox21 2025 re-screen loader (no network: replicate frames built inline)."""

import pandas as pd
import pytest

from cypshift.data.tox21 import AID, _clean, _max_response, load_auxiliary


def test_median_over_replicate_potencies_becomes_pic50():
    """AC50 is stored per replicate in micromolar; the median maps to pIC50."""
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Inactive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCC"],
            "Potency-Replicate_1": [1.0, None],
            "Potency-Replicate_2": [3.0, None],  # median of (1, 3) uM -> pIC50 6 - log10(2)
        }
    )
    cleaned = _clean(raw)
    assert set(cleaned["SMILES"]) == {"CCO"}  # the inactive has no potency and is dropped
    assert cleaned.set_index("SMILES").loc["CCO", "value"] == pytest.approx(6.0 - 0.3010, abs=1e-3)


def test_nonpositive_replicate_potencies_are_ignored():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"],
            "Potency-Replicate_1": [0.0],  # invalid, ignored
            "Potency-Replicate_2": [10.0],  # the only real reading -> pIC50 5.0
        }
    )
    assert _clean(raw).set_index("SMILES").loc["CCO", "value"] == pytest.approx(5.0)


def test_a_deposit_without_replicate_columns_raises():
    raw = pd.DataFrame(
        {"PUBCHEM_ACTIVITY_OUTCOME": ["Active"], "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"]}
    )
    with pytest.raises(KeyError, match="Potency-Replicate"):
        _clean(raw)


def test_uncovered_enzyme_is_empty(tmp_path):
    assert load_auxiliary("CYP2E1", pd.Series(["CCO"]), cache=tmp_path).empty


def test_reads_cache_and_deduplicates(tmp_path):
    (tmp_path / "tox21_CYP2D6.csv").write_text("SMILES,value\nCCO,6.0\nCCN,7.0\n")
    kept = load_auxiliary("CYP2D6", pd.Series(["OCC"]), cache=tmp_path)  # OCC == CCO canonical
    assert set(kept["SMILES"]) == {"CCN"}
    assert (kept["target"] == "CYP2D6").all()


def test_aid_map_covers_the_four_scored_enzymes():
    assert {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"} <= set(AID)


def test_max_response_is_the_negated_replicate_median_on_a_0_1_scale():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Inactive", "Inconclusive", "Inactive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCC", "CCCC", "CCCCC"],
            "Max_Response-Replicate_1": ["FLOAT", None, -40.0, 12.0],
            "Max_Response-Replicate_2": [-80.0, -10.0, -44.0, None],
            "Max_Response-Replicate_3": [-100.0, -20.0, -50.0, None],
        }
    )
    out = _max_response(raw).set_index("SMILES")
    assert set(out.index) == {"CCO", "CCC", "CCCCC"}  # inconclusive dropped
    assert out.loc["CCO", "value"] == pytest.approx(0.90)  # median of (-80, -100), text ignored
    assert out.loc["CCC", "value"] == pytest.approx(0.15)
    assert out.loc["CCCCC", "value"] == 0.0  # activation clips to zero
    assert not out["is_censored"].any()


def test_clean_routes_the_max_response_representation():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"],
            "Max_Response-Replicate_1": [-60.0],
            "Potency-Replicate_1": [1.0],
        }
    )
    assert _clean(raw, representation="max_response").loc[0, "value"] == pytest.approx(0.6)


def test_a_deposit_without_max_response_columns_raises():
    raw = pd.DataFrame(
        {"PUBCHEM_ACTIVITY_OUTCOME": ["Active"], "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"]}
    )
    with pytest.raises(KeyError, match="Max_Response-Replicate"):
        _max_response(raw)
