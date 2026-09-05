"""Tests for the ChEMBL auxiliary data loader."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.chembl import (
    PIC50_RANGE,
    POTENCY_TYPES,
    canonical,
    deduplicate_against,
    to_pic50,
)


def records(**overrides) -> pd.DataFrame:
    base = {
        "canonical_smiles": ["CCO", "c1ccccc1", "CCN", "CCCC"],
        "standard_value": [100.0, 1000.0, 50.0, 10000.0],
        "standard_units": ["nM"] * 4,
        "standard_relation": ["="] * 4,
        "standard_type": ["IC50"] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_potency_conversion_matches_the_definition():
    """pIC50 = 9 - log10(nM), so 100 nM is 7.0."""
    frame = to_pic50(records())
    value = frame.loc[frame["SMILES"] == "CCO", "value"].iloc[0]
    assert value == pytest.approx(7.0)


def test_censored_records_are_dropped_not_converted():
    """A '>10000 nM' reading says the compound is inactive. Converting it asserts a
    potency nobody measured."""
    frame = to_pic50(records(standard_relation=["=", ">", "<", "="]))
    assert set(frame["SMILES"]) == {"CCO", "CCCC"}


def test_non_nanomolar_units_are_dropped():
    frame = to_pic50(records(standard_units=["nM", "uM", "nM", "%"]))
    assert set(frame["SMILES"]) == {"CCO", "CCN"}


def test_values_outside_the_plausible_range_are_dropped():
    """Below the range is usually a unit error; above it is a detection limit."""
    low, high = PIC50_RANGE
    frame = to_pic50(records(standard_value=[100.0, 1e12, 1e-6, 50.0]))
    assert ((frame["value"] >= low) & (frame["value"] <= high)).all()


def test_replicates_are_aggregated_by_median():
    """ChEMBL pools many papers; a mean is pulled by one outlying assay."""
    frame = to_pic50(
        records(
            canonical_smiles=["CCO"] * 3 + ["CCN"],
            standard_value=[100.0, 100.0, 1.0, 50.0],
        )
    )
    assert frame.loc[frame["SMILES"] == "CCO", "value"].iloc[0] == pytest.approx(7.0)


def test_missing_smiles_or_value_are_dropped():
    frame = to_pic50(
        records(
            canonical_smiles=["CCO", None, "CCN", "CCCC"], standard_value=[100.0, 1.0, None, 10.0]
        )
    )
    assert set(frame["SMILES"]) == {"CCO", "CCCC"}


def test_an_empty_input_gives_an_empty_frame():
    assert to_pic50(pd.DataFrame()).empty


def test_canonical_smiles_match_across_spellings():
    """The same compound reaches the two sources written differently."""
    assert (
        canonical(pd.Series(["c1ccccc1"])).iloc[0] == canonical(pd.Series(["C1=CC=CC=C1"])).iloc[0]
    )


def test_unparseable_smiles_becomes_none():
    assert canonical(pd.Series(["not-a-molecule"])).iloc[0] is None


def test_challenge_compounds_are_removed():
    """Otherwise a compound is an auxiliary training row and a scored label at once,
    and the model is rewarded for memorising it."""
    auxiliary = pd.DataFrame({"SMILES": ["CCO", "CCN", "CCCC"], "value": [7.0, 6.0, 5.0]})
    kept = deduplicate_against(auxiliary, pd.Series(["C1=CC=CC=C1", "OCC"]))
    assert set(kept["SMILES"]) == {"CCN", "CCCC"}


def test_deduplication_matches_on_structure_not_string():
    """ "OCC" and "CCO" are the same molecule."""
    auxiliary = pd.DataFrame({"SMILES": ["CCO"], "value": [7.0]})
    assert deduplicate_against(auxiliary, pd.Series(["OCC"])).empty


def test_only_potency_types_are_requested():
    assert set(POTENCY_TYPES) == {"IC50", "Ki"}


def test_values_are_finite():
    assert np.isfinite(to_pic50(records())["value"]).all()
