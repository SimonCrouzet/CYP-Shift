"""The TDI submission calls: a threshold from the out-of-fold scores or from the test scores."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdi_submission as script  # noqa: E402 -- after the scripts path insert


def test_the_oof_rule_flags_the_prevalence_share_of_the_oof_scores_and_lets_the_test_share_float():
    oof = np.arange(100, dtype=float)
    truth = np.r_[np.zeros(80), np.ones(20)]  # prevalence 0.2
    higher_test = oof + 30
    called, threshold = script.calls(oof, truth, higher_test, "oof")
    assert threshold == pytest.approx(np.quantile(oof, 0.8))
    assert called.sum() > 20 and called[-1] and not called[0]


def test_the_share_rule_flags_the_prevalence_share_of_the_test_scores():
    oof = np.arange(100, dtype=float)
    truth = np.r_[np.zeros(75), np.ones(25)]
    test = np.arange(40, dtype=float) + 500
    called, _ = script.calls(oof, truth, test, "share")
    assert called.sum() == 10 and called[-10:].all()


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule"):
        script.calls(np.zeros(3), np.array([0, 1, 0]), np.zeros(3), "median")


def test_a_perfect_ranking_scores_one():
    truth = np.r_[np.zeros(8), np.ones(2)]
    report = script.out_of_fold_report(np.arange(10, dtype=float), truth)
    assert report == {"MCC": pytest.approx(1.0), "AUC": pytest.approx(1.0)}
