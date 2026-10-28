"""The stacking script: correction columns from the auxiliary heads, and the scored scenarios."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stack as script  # noqa: E402 -- after the scripts path insert


def _tables(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    names = [f"m{i}" for i in range(n)]
    truth = rng.normal(5.0, 1.0, n)
    head = rng.normal(size=n)  # an auxiliary head that carries the error of the direct prediction
    incumbent = pd.DataFrame({"Molecule_Name": names})
    auxiliary = pd.DataFrame({"Molecule_Name": names})
    for enzyme in script.ENZYMES:
        target = f"{enzyme}_pIC50_direct_inhibition"
        incumbent[target] = truth
        incumbent[f"{target}_conf_low"] = truth - 0.1
        incumbent[f"{target}_conf_high"] = truth + 0.1
        incumbent[f"{target}_pred"] = truth + 0.7 * head
        auxiliary[f"{target}__direct_inhibition_drc"] = 5.0
        auxiliary[f"{target}__luminescent_ac50"] = 5.0 + head  # minus the direct head: ``head``
        auxiliary[f"{target}__luminescent_max_response"] = rng.uniform(0, 1, n)
        auxiliary[f"{target}__fluorogenic_pic50"] = np.nan  # never trained for this enzyme
    folds = pd.Series(rng.integers(0, 5, n), index=names)
    return incumbent, auxiliary, folds


def test_potency_heads_enter_as_differences_from_the_direct_head_and_responses_as_they_are():
    _, auxiliary, _ = _tables(n=10)
    target = "CYP3A4_pIC50_direct_inhibition"
    columns = script.auxiliary_columns(auxiliary, target, pd.Index(auxiliary["Molecule_Name"]))
    assert sorted(columns) == ["luminescent_ac50", "luminescent_max_response"]
    direct = auxiliary[f"{target}__direct_inhibition_drc"].to_numpy()
    assert np.allclose(
        columns["luminescent_ac50"], auxiliary[f"{target}__luminescent_ac50"] - direct
    )
    assert np.allclose(
        columns["luminescent_max_response"], auxiliary[f"{target}__luminescent_max_response"]
    )


def test_an_informative_auxiliary_head_lowers_the_error_of_every_enzyme():
    incumbent, auxiliary, folds = _tables()
    table, weights = script.evaluate(incumbent, {}, auxiliary, folds)
    for enzyme in [*script.ENZYMES, "macro"]:
        rows = table[table["enzyme"] == enzyme]
        base = rows[rows["scenario"] == "incumbent"].iloc[0]
        raw = rows[(rows["scenario"] == "auxiliary") & ~rows["restored"]].iloc[0]
        assert raw["MAE"] < 0.5 * base["MAE"]
        assert raw["ST-RAE"] < 0.5 * base["ST-RAE"]
        assert raw["Pearson"] > base["Pearson"]
    assert set(weights) == set(script.ENZYMES)
    chosen = weights["CYP2D6"]["auxiliary"]
    assert chosen["columns"] == ["luminescent_ac50", "luminescent_max_response"]
    assert chosen["weights"][0] < -0.3  # the head is subtracted back out


def test_members_and_auxiliary_scenarios_are_scored_when_both_are_given():
    incumbent, auxiliary, folds = _tables()
    other = incumbent.copy()
    for enzyme in script.ENZYMES:
        other[f"{enzyme}_pIC50_direct_inhibition_pred"] += 0.05
    table, _ = script.evaluate(incumbent, {"other": other}, auxiliary, folds)
    assert set(table["scenario"]) == {"incumbent", "members", "auxiliary", "members+auxiliary"}
    assert set(table["enzyme"]) == {*script.ENZYMES, "macro"}


def test_a_compound_without_a_fold_is_an_error():
    incumbent, auxiliary, folds = _tables(n=20)
    with pytest.raises(ValueError, match="no fold"):
        script.evaluate(incumbent, {}, auxiliary, folds.iloc[:10])


def test_cells_are_read_from_the_run_bundles_and_blended(tmp_path, monkeypatch):
    incumbent, auxiliary, _ = _tables(n=12)
    for seed in (0, 1):
        bundle = tmp_path / "rev" / f"reg_cell_seed{seed}"
        bundle.mkdir(parents=True)
        shifted = incumbent.copy()
        for enzyme in script.ENZYMES:
            shifted[f"{enzyme}_pIC50_direct_inhibition_pred"] += seed
        shifted.to_parquet(bundle / "out_of_fold.parquet", index=False)
        auxiliary.to_parquet(bundle / "out_of_fold_auxiliary.parquet", index=False)
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    import export_cell_metrics

    monkeypatch.setattr(export_cell_metrics, "BUNDLES", tmp_path)
    blended = script.blend_of("rev", "cell@0+cell@1", 0)
    column = "CYP1A2_pIC50_direct_inhibition_pred"
    assert np.allclose(blended[column], incumbent[column] + 0.5)
    heads = script.read_auxiliary("rev", "cell@1", 0)
    assert list(heads.columns) == list(auxiliary.columns)
