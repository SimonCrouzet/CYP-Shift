"""Calibrating the single-concentration screen into usable pseudo-labels."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS
from cypshift.data.screen import (
    ScreenCalibrator,
    pseudo_labels,
    response_observations,
    to_wide,
)

RNG = np.random.default_rng(0)


def _screen_and_curves(n=400, promoted=200):
    """A screen where fold-change decreases with potency, and curves only for hits."""
    names = [f"OCNT-{i:05d}" for i in range(n)]
    smiles = ["CCO"] * n
    pic50 = RNG.uniform(2.5, 7.5, (n, len(ISOFORMS)))
    fold_change = -0.6 * pic50 + RNG.normal(0, 0.15, pic50.shape)

    rows = []
    for column, isoform in enumerate(ISOFORMS):
        for index, name in enumerate(names):
            rows.append(
                {
                    "Molecule_Name": name,
                    "SMILES": smiles[index],
                    "enzyme": isoform,
                    "log2fc_estimate": fold_change[index, column],
                }
            )
    screen = pd.DataFrame(rows)

    # Only the most potent compounds were promoted to a dose-response curve.
    order = np.argsort(-pic50[:, 0])[:promoted]
    curves = {"Molecule_Name": [names[i] for i in order], "SMILES": [smiles[i] for i in order]}
    for column, isoform in enumerate(ISOFORMS):
        curves[f"{isoform}_pIC50_direct_inhibition"] = pic50[order, column]
    return screen, pd.DataFrame(curves)


def test_response_observations_log2fc_keeps_the_raw_reading_unsquashed():
    """Tier C step 2: kind='log2fc' emits the raw measured log2 fold-change with a symmetric
    band in the same units, not the clipped [0,1] inhibited fraction."""
    screen, _ = _screen_and_curves(n=40, promoted=20)
    fraction = response_observations(screen, measurement_noise=0.1, kind="fraction")
    log2fc = response_observations(
        screen, measurement_noise=0.2, assay="direct_inhibition_screen_log2fc", kind="log2fc"
    )
    assert fraction["value"].between(0.0, 1.0).all()  # a squashed fraction
    # The raw reading equals the source column and is not confined to [0, 1].
    merged = log2fc.merge(
        screen.rename(columns={"enzyme": "target"}), on=["Molecule_Name", "target"]
    )
    assert np.allclose(merged["value"], merged["log2fc_estimate"])
    assert (log2fc["value"] < 0).any()  # potent compounds have negative fold-change
    assert np.allclose(log2fc["conf_high"] - log2fc["conf_low"], 0.4)  # 2 * noise, unclipped
    assert (log2fc["assay"] == "direct_inhibition_screen_log2fc").all()


def test_wide_form_has_one_column_per_enzyme():
    screen, _ = _screen_and_curves()
    wide = to_wide(screen)
    assert all(isoform in wide.columns for isoform in ISOFORMS)
    assert len(wide) == screen["Molecule_Name"].nunique()


def test_calibration_is_monotone_decreasing_in_fold_change():
    screen, curves = _screen_and_curves()
    calibrator = ScreenCalibrator().fit(to_wide(screen), curves)
    probe = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c"],
            "SMILES": ["CCO"] * 3,
            **{isoform: [-3.0, -1.5, 0.0] for isoform in ISOFORMS},
        }
    )
    predicted = calibrator.predict(probe)[f"{ISOFORMS[0]}_pIC50_direct_inhibition"]
    assert predicted.iloc[0] >= predicted.iloc[1] >= predicted.iloc[2]


def test_too_few_paired_observations_is_rejected():
    screen, curves = _screen_and_curves(n=60, promoted=10)
    with pytest.raises(ValueError, match="paired observations"):
        ScreenCalibrator().fit(to_wide(screen), curves)


def test_predicting_before_fitting_is_an_error():
    screen, _ = _screen_and_curves()
    with pytest.raises(RuntimeError, match="not fitted"):
        ScreenCalibrator().predict(to_wide(screen))


def test_readings_outside_the_fitted_range_are_marked_censored():
    screen, curves = _screen_and_curves()
    labels, _ = pseudo_labels(screen, curves, exclude_measured=False)
    isoform = ISOFORMS[0]
    assert labels[f"{isoform}_is_censored"].sum() > 0


def test_a_censored_label_carries_a_one_sided_bound():
    screen, curves = _screen_and_curves()
    labels, _ = pseudo_labels(screen, curves, exclude_measured=False)
    isoform = ISOFORMS[0]
    column = f"{isoform}_pIC50_direct_inhibition"
    censored = labels[labels[f"{isoform}_is_censored"]]
    # Bounded above by the calibration endpoint, unbounded below.
    assert (censored[f"{column}_conf_high"] == censored[column]).all()
    assert (censored[f"{column}_conf_low"] < censored[column] - 1).all()


def test_an_uncensored_label_carries_a_symmetric_band():
    screen, curves = _screen_and_curves()
    labels, noise = pseudo_labels(screen, curves, exclude_measured=False)
    isoform = ISOFORMS[0]
    column = f"{isoform}_pIC50_direct_inhibition"
    plain = labels[~labels[f"{isoform}_is_censored"]].dropna(subset=[column])
    width = plain[f"{column}_conf_high"] - plain[f"{column}_conf_low"]
    assert np.allclose(width, 2 * 1.96 * noise[isoform])


def test_pseudo_labels_never_overwrite_a_real_measurement():
    screen, curves = _screen_and_curves()
    labels, _ = pseudo_labels(screen, curves, exclude_measured=True)
    measured = set(curves["Molecule_Name"])
    column = f"{ISOFORMS[0]}_pIC50_direct_inhibition"
    overlapping = labels[labels["Molecule_Name"].isin(measured)]
    assert overlapping[column].isna().all()


def test_pseudo_labels_are_lower_than_the_curve_set():
    """The un-promoted compounds are the inactives, which is the whole point."""
    screen, curves = _screen_and_curves()
    labels, _ = pseudo_labels(screen, curves)
    column = f"{ISOFORMS[0]}_pIC50_direct_inhibition"
    assert labels[column].median() < curves[column].median()
