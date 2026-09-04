"""Assembling measurements into one conditioned observation table."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS
from cypshift.data.assembly import (
    ASSAYS,
    OBSERVATION_COLUMNS,
    SourceWeights,
    build_observations,
    one_hot_encode,
)


def _inhibition(n=20):
    frame = {"Molecule_Name": [f"OCNT-{i:05d}" for i in range(n)], "SMILES": ["CCO"] * n}
    for isoform in ISOFORMS:
        y = np.linspace(3.0, 7.0, n)
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    return pd.DataFrame(frame)


def _screen_labels(n=15):
    frame = {"Molecule_Name": [f"SCRN-{i:05d}" for i in range(n)], "SMILES": ["CCC"] * n}
    for isoform in ISOFORMS:
        y = np.linspace(2.0, 4.0, n)
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 1.0
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y
        frame[f"{isoform}_is_censored"] = True
    return pd.DataFrame(frame)


def test_observations_carry_the_declared_columns():
    observations = build_observations(_inhibition())
    assert list(observations.columns) == list(OBSERVATION_COLUMNS)


def test_every_measured_cell_becomes_one_row():
    inhibition = _inhibition(n=20)
    observations = build_observations(inhibition)
    assert len(observations) == 20 * len(ISOFORMS)


def test_sources_are_labelled_by_assay_and_weighted_differently():
    observations = build_observations(_inhibition(), screen_labels=_screen_labels())
    by_assay = observations.groupby("assay")["weight"].first()
    assert by_assay["direct_inhibition_drc"] > by_assay["direct_inhibition_screen"]
    assert set(observations["assay"]) == {"direct_inhibition_drc", "direct_inhibition_screen"}


def test_screen_rows_keep_their_censored_flag():
    observations = build_observations(_inhibition(), screen_labels=_screen_labels())
    screen_rows = observations[observations["assay"] == "direct_inhibition_screen"]
    assert screen_rows["is_censored"].all()
    assert not observations[observations["assay"] == "direct_inhibition_drc"]["is_censored"].any()


def test_credible_bounds_survive_the_melt():
    observations = build_observations(_inhibition())
    assert np.allclose(observations["conf_high"] - observations["conf_low"], 0.4)


def test_custom_weights_are_respected():
    weights = SourceWeights(dose_response=2.0, screen_pseudo_label=0.1)
    observations = build_observations(_inhibition(), _screen_labels(), weights=weights)
    by_assay = observations.groupby("assay")["weight"].first()
    assert by_assay["direct_inhibition_drc"] == 2.0
    assert by_assay["direct_inhibition_screen"] == 0.1


def test_conditioning_is_one_hot_over_targets_and_assays():
    observations = build_observations(_inhibition(), _screen_labels())
    one_hot = one_hot_encode(observations)
    assert one_hot.shape == (len(observations), len(ISOFORMS) + len(ASSAYS))
    # Exactly one target and exactly one assay set per row.
    assert np.all(one_hot[:, : len(ISOFORMS)].sum(axis=1) == 1)
    assert np.all(one_hot[:, len(ISOFORMS) :].sum(axis=1) == 1)


def test_an_unrecognised_target_is_rejected():
    observations = build_observations(_inhibition())
    observations.loc[0, "target"] = "CYP2C19"
    with pytest.raises(ValueError, match="unrecognised target"):
        one_hot_encode(observations)


def test_the_same_compound_on_two_targets_differs_only_in_conditioning():
    observations = build_observations(_inhibition(n=5))
    one_hot = one_hot_encode(observations)
    same_compound = observations["Molecule_Name"] == "OCNT-00000"
    rows = one_hot[same_compound.to_numpy()]
    assert len(rows) == len(ISOFORMS)
    assert len({tuple(row) for row in rows}) == len(ISOFORMS)
