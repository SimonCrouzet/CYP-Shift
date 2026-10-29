"""The TDI baseline stacks a graph run's saved heads and encodings, keyed by compound."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdi_baseline as script  # noqa: E402 -- after the scripts path insert

ENDPOINTS = [f"{e}_pIC50_direct_inhibition" for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")]


def _bundle(tmp_path):
    names = ["A", "B"]
    direct = pd.DataFrame({"Molecule_Name": names})
    heads = pd.DataFrame({"Molecule_Name": names})
    for endpoint in ENDPOINTS:
        direct[f"{endpoint}_pred"] = [4.0, 5.0]
    heads["CYP3A4_pIC50_direct_inhibition__tdi_drc"] = [4.5, 5.0]
    direct.to_parquet(tmp_path / "out_of_fold.parquet")
    heads.to_parquet(tmp_path / "out_of_fold_auxiliary.parquet")
    pd.DataFrame({"Molecule_Name": names, "e0": [1.0, 2.0], "e1": [3.0, 4.0]}).to_parquet(
        tmp_path / "out_of_fold_embedding.parquet"
    )
    return tmp_path


def test_the_heads_are_stacked_with_the_preincubated_shift_and_cover_the_run_s_compounds(tmp_path):
    table = pd.DataFrame({"Molecule_Name": ["B", "Z"]})  # Z is not in the run's folds
    features, covered = script.bundle_features(table, _bundle(tmp_path), embedding=False)
    assert covered.tolist() == [True, False]
    assert features.shape == (2, 6)  # four direct predictions, the head, the shift
    assert features[0, -1] == 0.0 and np.isnan(features[1]).all()  # shift = 5.0 - 5.0


def test_the_encodings_are_read_per_compound(tmp_path):
    table = pd.DataFrame({"Molecule_Name": ["B", "A"]})
    features, covered = script.bundle_features(table, _bundle(tmp_path), embedding=True)
    assert features.tolist() == [[2.0, 4.0], [1.0, 3.0]] and covered.all()


def test_only_the_scored_compounds_get_a_score_while_every_labelled_one_trains():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 4))
    labels = (features[:, 0] > 0).astype(float)
    labels[[3, 100]] = np.nan
    folds = np.arange(120) % 2
    scored = np.arange(120) >= 60
    scores, test_scores = script.out_of_fold_scores(
        features, labels, folds, threads=1, seed=0, scored=scored
    )
    assert test_scores is None
    assert np.isnan(scores[:60]).all() and np.isnan(scores[100])
    assert not np.isnan(np.delete(scores[60:], 40)).any()


def test_neighbour_scores_are_the_similarity_weighted_label_of_the_nearest_compounds():
    train = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    labels = np.array([1, 1, 0, 0])
    test = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    scores = script.neighbour_scores(train, labels, test, neighbours=2)
    assert scores.tolist() == [1.0, 0.0, 0.5]  # an all-zero fingerprint falls back to the mean


def test_the_fold_models_also_score_the_test_features_and_a_timing_run_stops_early():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(90, 3))
    labels = (features[:, 0] > 0).astype(float)
    folds = np.arange(90) % 3
    scores, test_scores = script.out_of_fold_scores(
        features, labels, folds, threads=1, seed=0, test_features=features[:7], max_folds=2
    )
    assert test_scores.shape == (7,) and ((test_scores >= 0) & (test_scores <= 1)).all()
    assert np.isnan(scores[folds == 2]).all() and not np.isnan(scores[folds < 2]).any()


def _noise_table():
    """Four compounds: clear positive, ambiguous, clear negative, and an inferred positive."""
    return pd.DataFrame(
        {
            "CYP2D6_pIC50_direct_inhibition": [5.0, 5.0, 5.0, 3.5],
            "CYP2D6_pIC50_TDI_condition": [6.0, 5.32, 5.0, 4.6],
            "CYP2D6_pIC50_direct_inhibition_std": [0.05, 0.05, 0.05, 0.05],
            "CYP2D6_pIC50_TDI_condition_std": [0.05, 0.05, 0.05, 0.05],
        }
    )


def test_a_label_is_read_from_a_measured_shift_only_above_the_direct_floor():
    shift, z, error = script.shift_noise(_noise_table(), "CYP2D6")
    assert shift[:3].tolist() == pytest.approx([1.0, 0.32, 0.0])
    assert np.isnan(shift[3]) and np.isnan(z[3]) and np.isnan(error[3])  # inferred, direct < 4
    assert z[0] > 5 and abs(z[1]) < 1 < abs(z[2])
    assert error[0] == pytest.approx(np.sqrt(2) * 0.05)


def test_the_label_modes_leave_out_down_weight_or_replace_the_ambiguous_labels():
    shift, z, error = script.shift_noise(_noise_table(), "CYP2D6")
    labels = np.array([1.0, 1.0, 0.0, 1.0])
    same, weights, regression = script.label_mode_inputs("hard", labels, shift, z, error)
    assert same is labels and weights is None and not regression
    dropped, weights, _ = script.label_mode_inputs("drop_ambiguous", labels, shift, z, error)
    assert np.isnan(dropped[1]) and dropped[[0, 2, 3]].tolist() == [1.0, 0.0, 1.0]
    kept, weights, _ = script.label_mode_inputs("soft", labels, shift, z, error)
    assert weights[0] == 1.0 and weights[3] == 1.0  # clear and inferred labels keep full weight
    assert weights[1] < 0.5 < weights[2] and kept is labels
    target, weights, regression = script.label_mode_inputs("shift", labels, shift, z, error)
    assert regression and target[:3].tolist() == pytest.approx([1.0, 0.32, 0.0])
    assert np.isnan(target[3])  # no measured shift to regress
    with pytest.raises(ValueError, match="label mode"):
        script.label_mode_inputs("median", labels, shift, z, error)


def test_a_regression_scores_by_the_predicted_value_with_weights_and_only_for_lightgbm():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(120, 3))
    target = features[:, 0] * 2.0
    scores = script.fold_scores(
        "lightgbm",
        features[:90],
        target[:90],
        features[90:],
        1,
        0,
        weights=np.ones(90),
        regression=True,
    )
    assert np.corrcoef(scores, target[90:])[0, 1] > 0.7
    with pytest.raises(ValueError, match="LightGBM"):
        script.fold_scores("knn", features[:90], target[:90], features[90:], 1, 0, regression=True)


def test_the_live_prevalence_reweighting_lowers_the_mcc_of_a_ranking_with_many_false_calls():
    rng = np.random.default_rng(3)
    truth = (rng.random(2000) < 0.3).astype(int)
    score = truth + rng.normal(scale=1.2, size=2000)
    plain = script.mcc_at_prevalence(truth, score, truth.mean())
    live = script.reweighted_mcc(truth, score, 0.07, 0.07)
    assert 0 < live < plain
    assert script.reweighted_mcc(truth, score, truth.mean(), truth.mean()) == pytest.approx(
        plain, abs=0.02
    )
