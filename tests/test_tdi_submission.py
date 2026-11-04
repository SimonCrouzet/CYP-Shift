"""The TDI submission: members on different scales, one blend per enzyme, and the call rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdi_submission as script  # noqa: E402 -- after the scripts path insert

COLUMN = script.label_column("CYP3A4")


def test_the_oof_rule_flags_the_prevalence_share_of_the_oof_scores_and_lets_the_test_share_float():
    oof = np.arange(100, dtype=float)
    truth = np.r_[np.zeros(80), np.ones(20)]  # prevalence 0.2
    called, threshold = script.calls(oof, truth, oof + 30, "oof")
    assert threshold == pytest.approx(np.quantile(oof, 0.8))
    assert called.sum() > 20 and called[-1] and not called[0]


def test_the_share_rule_flags_the_prevalence_share_of_the_test_scores():
    truth = np.r_[np.zeros(75), np.ones(25)]
    test = np.arange(40, dtype=float) + 500
    called, _ = script.calls(np.arange(100, dtype=float), truth, test, "share")
    assert called.sum() == 10 and called[-10:].all()


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule"):
        script.calls(np.zeros(3), np.array([0, 1, 0]), np.zeros(3), "median")


def test_a_perfect_ranking_scores_one_overall_and_within_a_subset():
    truth = np.r_[np.zeros(8), np.ones(2)]
    score = np.arange(10, dtype=float)
    assert script.out_of_fold_report(score, truth) == {
        "n": 10,
        "positives": 2,
        "MCC": pytest.approx(1.0),
        "AUC": pytest.approx(1.0),
    }
    subset = np.arange(10) >= 5
    report = script.out_of_fold_report(score, truth, subset)
    assert report["n"] == 5 and report["positives"] == 2 and report["AUC"] == pytest.approx(1.0)
    only_negatives = np.arange(10) < 5
    assert "AUC" not in script.out_of_fold_report(score, truth, only_negatives)


def test_members_are_read_from_their_specs():
    graph = script.parse_member("graph:cell_a@0,2")
    assert (graph.kind, graph.name, graph.seeds) == ("graph", "cell_a", (0, 2))
    assert script.parse_member("graph:cell_a").seeds == (0,)
    table = script.parse_member("table:oof.csv,test.csv")
    assert (table.oof_path, table.test_path) == (Path("oof.csv"), Path("test.csv"))
    for bad in ("graph:", "table:only_one.csv", "forest:x"):
        with pytest.raises(ValueError, match="member"):
            script.parse_member(bad)
    members = script.parse_members(["CYP2D6=graph:a+graph:b"])
    assert [m.name for m in members["CYP2D6"]] == ["a", "b"]
    with pytest.raises(ValueError, match="ENZYME"):
        script.parse_members(["CYP1A2=graph:a"])


def test_members_on_different_scales_blend_by_their_place_among_their_own_oof_scores():
    oof = np.arange(10, dtype=float)
    # the second member's scale is 1000 times larger and its test scores are shifted
    blended_oof, blended_test = script.blend(
        [oof, oof * 1000], [np.array([-1.0, 4.5, 20.0]), np.array([-1.0, 4500.0, 20000.0])]
    )
    assert blended_oof.tolist() == pytest.approx((np.arange(10) + 1) / 10)
    assert blended_test.tolist() == pytest.approx([0.0, 0.5, 1.0])


def _write_bundle(root: Path, name: str, seed: int, offset: float) -> None:
    path = root / "rev" / f"reg_{name}_seed{seed}"
    path.mkdir(parents=True)
    pd.DataFrame({"Molecule_Name": ["a", "b"], COLUMN: [0.1 + offset, 0.9 + offset]}).to_parquet(
        path / "out_of_fold_auxiliary.parquet"
    )
    test = pd.DataFrame(
        {"Molecule_Name": ["t", "t", "u", "u"], "fold": [0, 1, 0, 1], COLUMN: [0, 1, 2, 4.0]}
    )
    test[COLUMN] += offset
    test.to_parquet(path / "test_by_fold_auxiliary.parquet")


def test_a_graph_member_averages_the_fold_models_and_the_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "BUNDLES", tmp_path)
    _write_bundle(tmp_path, "cell", 0, 0.0)
    _write_bundle(tmp_path, "cell", 1, 1.0)
    member = script.parse_member("graph:cell@0,1")
    oof, test = script.member_scores(member, "CYP3A4", "rev")
    assert oof.to_dict() == pytest.approx({"a": 0.6, "b": 1.4})
    assert test.to_dict() == pytest.approx({"t": 1.0, "u": 3.5})


def test_a_table_member_reads_the_baseline_s_score_columns(tmp_path):
    pd.DataFrame({"Molecule_Name": ["a", "b", "c"], "CYP3A4_score": [0.2, np.nan, 0.7]}).to_csv(
        tmp_path / "oof.csv", index=False
    )
    pd.DataFrame({"Molecule_Name": ["t"], "CYP3A4_score": [0.5]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    member = script.parse_member(f"table:{tmp_path / 'oof.csv'},{tmp_path / 'test.csv'}")
    oof, test = script.member_scores(member, "CYP3A4", "rev")
    assert oof.to_dict() == {"a": 0.2, "c": 0.7} and test.to_dict() == {"t": 0.5}
