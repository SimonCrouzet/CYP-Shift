"""Tests for the Novartis CYP3A4 TDI loader (cache-based; no Polaris, no network)."""

import pandas as pd
import pytest

from cypshift.data.novartis import NOVARTIS_ENZYME, load


def _cache(tmp_path, rows):
    path = tmp_path / "cyp3a4_tdi.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_is_tdi_is_thresholded_on_log_kobs(tmp_path):
    cache = _cache(tmp_path, {"SMILES": ["CCN", "CCC"], "log_kobs": [-1.0, -3.0]})
    out = load(pd.Series(["X"]), cache=cache, threshold=-2.0).set_index("SMILES")
    assert bool(out.loc["CCN", "is_TDI"]) is True  # -1.0 > -2.0
    assert bool(out.loc["CCC", "is_TDI"]) is False  # -3.0 < -2.0


def test_challenge_compounds_are_removed(tmp_path):
    cache = _cache(tmp_path, {"SMILES": ["CCO", "CCN"], "log_kobs": [-1.0, -1.0]})
    kept = load(pd.Series(["OCC"]), cache=cache)  # OCC == CCO canonical
    assert set(kept["SMILES"]) == {"CCN"}
    assert (kept["target"] == NOVARTIS_ENZYME).all()


def test_missing_sdk_and_cache_raises(tmp_path):
    # No cache present and Polaris almost certainly absent: a clear error, not a silent pass.
    with pytest.raises((ImportError, FileNotFoundError, Exception)):
        load(pd.Series(["X"]), cache=tmp_path / "absent.csv")
