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


def test_detection_limit_read_from_concentration_columns():
    """The limit is computed from the deposit's top 'Activity at <conc> uM' column."""
    import numpy as np

    from cypshift.data.veith import detection_limit_pic50

    raw = pd.DataFrame(columns=["Activity at 0.457 uM", "Activity at 57.14 uM", "Potency"])
    assert detection_limit_pic50(raw) == pytest.approx(6.0 - np.log10(57.14))
    assert detection_limit_pic50(pd.DataFrame(columns=["Potency"])) is None


def test_inactives_kept_as_censored_when_requested():
    """include_inactives keeps inactives as one-sided rows below the data-derived limit."""
    import numpy as np

    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Inactive", "Inactive", "Inconclusive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCC", "CCCC", "CCCCC"],
            "Potency": [1.0, None, None, None],
            "Curve_Description": ["Complete curve; high efficacy", "", "", ""],
            "Activity at 0.457 uM": [40, 0, 0, 0],
            "Activity at 57.14 uM": [90, 2, 1, 0],
        }
    )
    limit = 6.0 - np.log10(57.14)
    off = _clean(raw, include_inactives=False)
    assert set(off["SMILES"]) == {"CCO"} and not off["is_censored"].any()

    on = _clean(raw, include_inactives=True).set_index("SMILES")
    assert set(on.index) == {"CCO", "CCC", "CCCC"}  # the inconclusive compound is dropped
    assert not on.loc["CCO", "is_censored"]
    assert on.loc["CCC", "is_censored"]
    assert on.loc["CCC", "value"] == pytest.approx(limit)
    assert on.loc["CCC", "conf_high"] == pytest.approx(limit)


def test_max_response_keeps_all_screened_on_a_0_1_scale():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Inactive", "Inconclusive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCC", "CCCC"],
            "Max_Response": [95.0, 3.0, 50.0],
            "Potency": [1.0, None, None],
        }
    )
    out = _clean(raw, representation="max_response").set_index("SMILES")
    assert set(out.index) == {"CCO", "CCC"}  # inconclusive dropped, inactive kept
    assert out.loc["CCO", "value"] == pytest.approx(0.95)
    assert out.loc["CCC", "value"] == pytest.approx(0.03)
    assert not out["is_censored"].any()


def test_max_response_needs_its_column():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO"],
            "Potency": [1.0],
        }
    )
    with pytest.raises(KeyError, match="Max_Response"):
        _clean(raw, representation="max_response")


def test_keeping_inactives_without_concentration_columns_raises():
    raw = pd.DataFrame(
        {
            "PUBCHEM_ACTIVITY_OUTCOME": ["Active", "Inactive"],
            "PUBCHEM_EXT_DATASOURCE_SMILES": ["CCO", "CCC"],
            "Potency": [1.0, None],
            "Curve_Description": ["Complete curve; high efficacy", ""],
        }
    )
    with pytest.raises(KeyError, match="Activity at"):
        _clean(raw, include_inactives=True)
