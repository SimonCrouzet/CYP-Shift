"""Tests for the Octant CYP3A4 auxiliary loader."""

import pandas as pd
import pytest

from cypshift.data.octant import OCTANT_ENZYME, load_auxiliary

_COLS = [
    "ocnt_batch",
    "CYP3A4_pIC50",
    "CYP3A4_pIC50_ci_lower",
    "CYP3A4_pIC50_ci_upper",
    "drc_qc_status",
    "standardized_smiles",
]
TSV = "\t".join(_COLS) + "\n"


def _write(tmp_path, body):
    p = tmp_path / "inhibition.tsv"
    p.write_text(TSV + body)
    return p


def test_only_cyp3a4_is_covered(tmp_path):
    tsv = _write(tmp_path, "OCNT-1\t6.0\t5.8\t6.2\tPASS\tCCO\n")
    assert load_auxiliary("CYP2D6", pd.Series(["X"]), cache=tsv).empty
    assert len(load_auxiliary("CYP3A4", pd.Series(["X"]), cache=tsv)) == 1


def test_credible_interval_is_kept(tmp_path):
    tsv = _write(tmp_path, "OCNT-1\t6.0\t5.8\t6.2\tPASS\tCCO\n")
    row = load_auxiliary("CYP3A4", pd.Series(["X"]), cache=tsv).iloc[0]
    assert row["conf_low"] == pytest.approx(5.8)
    assert row["conf_high"] == pytest.approx(6.2)


def test_failed_qc_rows_are_dropped(tmp_path):
    tsv = _write(tmp_path, "OCNT-1\t6.0\t5.8\t6.2\tPASS\tCCO\nOCNT-2\t7.0\t6.8\t7.2\tFAIL\tCCN\n")
    kept = load_auxiliary("CYP3A4", pd.Series(["X"]), cache=tsv)
    assert set(kept["SMILES"]) == {"CCO"}


def test_challenge_compounds_are_removed(tmp_path):
    tsv = _write(tmp_path, "OCNT-1\t6.0\t5.8\t6.2\tPASS\tCCO\nOCNT-2\t7.0\t6.8\t7.2\tPASS\tCCN\n")
    kept = load_auxiliary("CYP3A4", pd.Series(["OCC"]), cache=tsv)  # OCC == CCO canonical
    assert set(kept["SMILES"]) == {"CCN"}


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Octant"):
        load_auxiliary("CYP3A4", pd.Series(["X"]), cache=tmp_path / "absent.tsv")


def test_octant_enzyme_constant():
    assert OCTANT_ENZYME == "CYP3A4"
