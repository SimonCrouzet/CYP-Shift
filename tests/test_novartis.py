"""Tests for the Novartis loader (parquet cache; no Polaris, no network)."""

import pandas as pd
import pytest

from cypshift.data.novartis import NOVARTIS_ENZYMES, load_auxiliary, load_tdi


def _cache(tmp_path):
    frame = pd.DataFrame(
        {
            "MOL_smiles": ["CCO", "CCN", "CCC"],
            "pIC50_CYP3A4": [6.0, 5.0, 4.5],
            "pIC50_CYP2C9": [6.1, 5.1, 4.6],
            "pIC50_CYP2D6": [6.2, 5.2, 4.7],
            "log_kobs": [-1.0, -2.5, -3.0],
            "CLS_log_kobs": [2.0, 1.0, 0.0],
        }
    )
    path = tmp_path / "cyp3a4_v1.csv"
    frame.to_csv(path, index=False)
    return path


def test_reversible_pic50_is_returned_per_covered_enzyme(tmp_path):
    cache = _cache(tmp_path)
    rows = load_auxiliary("CYP2D6", pd.Series(["X"]), cache=cache)
    assert set(rows["SMILES"]) == {"CCO", "CCN", "CCC"}
    assert (rows["target"] == "CYP2D6").all()
    assert rows.set_index("SMILES").loc["CCO", "value"] == pytest.approx(6.2)


def test_uncovered_enzyme_is_empty(tmp_path):
    assert load_auxiliary("CYP1A2", pd.Series(["X"]), cache=_cache(tmp_path)).empty


def test_challenge_compounds_are_removed(tmp_path):
    kept = load_auxiliary("CYP3A4", pd.Series(["OCC"]), cache=_cache(tmp_path))  # OCC == CCO
    assert "CCO" not in set(kept["SMILES"])


def test_tdi_columns_are_exposed_separately(tmp_path):
    tdi = load_tdi(pd.Series(["X"]), cache=_cache(tmp_path))
    assert set(tdi.columns) >= {"SMILES", "log_kobs", "tdi_class"}
    assert len(tdi) == 3


def test_covered_enzymes_are_the_three_with_a_pic50_column():
    assert NOVARTIS_ENZYMES == ("CYP3A4", "CYP2C9", "CYP2D6")
