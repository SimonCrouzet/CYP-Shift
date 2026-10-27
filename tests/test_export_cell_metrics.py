"""The cell metrics are computed per enzyme on the measured compounds, then averaged."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_cell_metrics as script  # noqa: E402 -- after the scripts path insert


def _oof() -> pd.DataFrame:
    truth = np.array([4.0, 5.0, 6.0, 7.0])
    columns = {"Molecule_Name": [f"m{i}" for i in range(4)]}
    for enzyme in script.ENZYMES:
        target = f"{enzyme}_pIC50_direct_inhibition"
        columns[target] = truth
        columns[f"{target}_conf_low"] = truth - 0.1
        columns[f"{target}_conf_high"] = truth + 0.1
        columns[f"{target}_pred"] = truth + 0.5
    return pd.DataFrame(columns)


def test_a_constant_offset_keeps_correlation_one_and_sets_mae_to_the_offset():
    metrics = script.bundle_metrics(_oof())
    assert metrics["Pearson"] == pytest.approx(1.0)
    assert metrics["Spearman"] == pytest.approx(1.0)
    assert metrics["MAE"] == pytest.approx(0.5)


def test_compounds_without_a_measurement_for_an_enzyme_are_left_out_of_that_enzyme():
    oof = _oof()
    oof.loc[0, "CYP1A2_pIC50_direct_inhibition"] = np.nan
    oof.loc[0, "CYP1A2_pIC50_direct_inhibition_pred"] = 99.0  # would wreck the MAE if kept
    assert script.bundle_metrics(oof)["MAE"] == pytest.approx(0.5)


def test_cell_metrics_reads_the_bundles_by_recipe_name(tmp_path, monkeypatch):
    bundle = tmp_path / "rev" / "reg_cell_a_seed0"
    bundle.mkdir(parents=True)
    _oof().to_parquet(bundle / "out_of_fold.parquet")
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.cell_metrics("rev", ["cell_a"])
    assert table["name"].tolist() == ["cell_a"]
    assert list(table.columns) == ["name", *script.COLUMNS]


def test_the_per_enzyme_rows_average_to_the_cell_row():
    per_enzyme = script.enzyme_metrics(_oof())
    assert per_enzyme["enzyme"].tolist() == list(script.ENZYMES)
    assert per_enzyme["MAE"].mean() == pytest.approx(script.bundle_metrics(_oof())["MAE"])


def test_a_blend_averages_the_predictions_of_its_cells(tmp_path, monkeypatch):
    low, high = _oof(), _oof()
    for column in (c for c in low.columns if c.endswith("_pred")):
        low[column] = low[column] - 0.5  # exact predictions
        high[column] = high[column] + 0.5  # off by 1
    for name, table in (("low", low), ("high", high)):
        bundle = tmp_path / "rev" / f"reg_{name}_seed0"
        bundle.mkdir(parents=True)
        table.to_parquet(bundle / "out_of_fold.parquet")
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    table = script.cell_metrics("rev", ["low"], blends={"both": ["low", "rev/high"]})
    rows = table.set_index("name")["MAE"]
    assert rows["low"] == pytest.approx(0.0) and rows["both"] == pytest.approx(0.5)
