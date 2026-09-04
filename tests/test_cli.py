"""The CLI must expose the documented commands and score a well-formed submission."""

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from cypshift.cli import app
from cypshift.config import REGRESSION_ENDPOINTS

runner = CliRunner()


def _truth_and_predictions(tmp_path, n=40):
    rng = np.random.default_rng(3)
    names = [f"OCNT-{i:05d}" for i in range(n)]
    truth = {"Molecule_Name": names}
    preds = {"Molecule_Name": names}
    for endpoint in REGRESSION_ENDPOINTS:
        y = rng.uniform(3, 7, n)
        truth[endpoint] = y
        truth[f"{endpoint}_conf_low"] = y - 0.2
        truth[f"{endpoint}_conf_high"] = y + 0.2
        preds[endpoint] = y + rng.normal(0, 0.4, n)
    truth_path = tmp_path / "truth.csv"
    preds_path = tmp_path / "preds.csv"
    pd.DataFrame(truth).to_csv(truth_path, index=False)
    pd.DataFrame(preds).to_csv(preds_path, index=False)
    return truth_path, preds_path


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("fetch", "score", "reference"):
        assert command in result.stdout


def test_score_reports_every_endpoint_and_the_macro(tmp_path):
    truth_path, preds_path = _truth_and_predictions(tmp_path)
    result = runner.invoke(app, ["score", str(preds_path), "--truth", str(truth_path)])
    assert result.exit_code == 0, result.stdout
    assert "ST-RAE_mean" in result.stdout
    assert "MA" in result.stdout


def test_score_rejects_a_submission_with_a_missing_prediction(tmp_path):
    truth_path, preds_path = _truth_and_predictions(tmp_path)
    frame = pd.read_csv(preds_path)
    frame.loc[0, REGRESSION_ENDPOINTS[0]] = np.nan
    frame.to_csv(preds_path, index=False)
    result = runner.invoke(app, ["score", str(preds_path), "--truth", str(truth_path)])
    assert result.exit_code != 0
