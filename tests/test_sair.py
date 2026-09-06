"""Tests for the SAIR auxiliary loader (no parquet needed -- clean_rows and dedup)."""

import pandas as pd
import pytest

from cypshift.data.chembl import PIC50_RANGE
from cypshift.data.sair import clean_rows, load_auxiliary


def _rows(**overrides) -> pd.DataFrame:
    base = {
        "SMILES": ["CCO", "CCO", "c1ccccc1", "CCN"],
        "pIC50": [5.0, 7.0, 6.0, 8.0],
        "protein": ["P10635"] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_rows_are_on_the_pic50_scale_already():
    cleaned = clean_rows(_rows())
    assert set(cleaned.columns) == {"SMILES", "value"}
    assert cleaned["value"].between(*PIC50_RANGE).all()


def test_replicates_are_aggregated_by_median():
    # CCO appears at 5.0 and 7.0 -> median 6.0
    cleaned = clean_rows(_rows())
    assert cleaned.loc[cleaned["SMILES"] == "CCO", "value"].iloc[0] == pytest.approx(6.0)


def test_values_outside_the_range_are_dropped():
    low, high = PIC50_RANGE
    cleaned = clean_rows(
        _rows(pIC50=[low - 1, high + 1, 6.0, 8.0], SMILES=["A", "B", "c1ccccc1", "CCN"])
    )
    assert set(cleaned["SMILES"]) == {"c1ccccc1", "CCN"}


def test_empty_input_gives_empty_frame():
    assert clean_rows(pd.DataFrame()).empty


def test_missing_values_are_dropped():
    cleaned = clean_rows(_rows(pIC50=[5.0, None, 6.0, 8.0], SMILES=["CCO", "X", "c1ccccc1", "CCN"]))
    assert "X" not in set(cleaned["SMILES"])


def test_unknown_enzyme_is_refused():
    with pytest.raises(KeyError, match="pinned"):
        load_auxiliary("CYP2E9", pd.Series(["CCO"]))


def test_missing_index_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="SAIR index"):
        load_auxiliary(
            "CYP2D6", pd.Series(["CCO"]), cache=tmp_path, index=tmp_path / "absent.parquet"
        )
