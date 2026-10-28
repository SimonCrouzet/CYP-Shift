"""The learning-curve plot reads its table off the mean curves and styles variants as dashed."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_learning_curves as script  # noqa: E402 -- after the scripts path insert


def _curves() -> pd.DataFrame:
    rows = []
    for fold, errors in enumerate([[0.9, 0.6, 0.7], [0.8, 0.8, 0.5]]):
        for epoch, error in enumerate(errors, start=1):
            rows.append(
                {
                    "name": "a",
                    "fold": fold,
                    "epoch": epoch,
                    "fold_mae": error,
                    "train_loss": 1 / epoch,
                }
            )
    return pd.DataFrame(rows)


def test_curve_summary_reads_the_lowest_point_of_the_mean_curve_not_each_folds_own():
    summary = script.curve_summary(_curves(), "a")
    # mean curve 0.85, 0.70, 0.60: lowest at the last epoch, where the folds' own minima differ
    assert summary["epoch"] == 3
    assert summary["lowest"] == pytest.approx(0.60)
    assert summary["last"] == pytest.approx(0.60)
    assert summary["loss"] == pytest.approx(1 / 3)


def test_a_variant_takes_its_parents_colour_and_is_dashed():
    styles = script.assign_styles(["base", "base_100", "wide"], {"base_100": "base"})
    assert styles["base_100"][0] == styles["base"][0]
    assert styles["base_100"][1] == script.DASH
    assert styles["base"][1] == "-"
    assert styles["wide"][0] == script.SERIES[1]


def test_a_variant_of_an_unplotted_cell_is_rejected():
    with pytest.raises(ValueError, match="parent"):
        script.assign_styles(["a"], {"b": "missing"})


def test_more_parents_than_palette_slots_are_rejected():
    with pytest.raises(ValueError, match="at most"):
        script.assign_styles([f"cell{i}" for i in range(len(script.SERIES) + 1)], {})


def test_the_variants_of_one_parent_take_different_dash_patterns():
    styles = script.assign_styles(["base", "a", "b"], {"a": "base", "b": "base"})
    assert styles["a"][0] == styles["b"][0] == styles["base"][0]
    assert styles["a"][1] == script.DASH and styles["b"][1] != styles["a"][1]
