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
    for command in ("fetch", "score", "reference", "baseline"):
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


class TestSubmit:
    """The submission path: fit on everything, predict the blinded set, write a CSV."""

    def test_submit_is_registered(self):
        result = CliRunner().invoke(app, ["--help"], env={"COLUMNS": "200"})
        assert "submit" in result.output

    def test_submit_help_describes_the_blinded_set(self):
        result = CliRunner().invoke(app, ["submit", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0
        assert "blinded" in result.output.lower()

    def test_a_written_submission_has_the_scored_columns(self, tmp_path, monkeypatch):
        """Every regression endpoint must be present and complete, or the leaderboard
        rejects the file."""
        import numpy as np
        import pandas as pd

        from cypshift.config import REGRESSION_ENDPOINTS

        test_frame = pd.DataFrame(
            {"Molecule_Name": ["A", "B", "C"], "SMILES": ["CCO", "c1ccccc1", "CCN"]}
        )
        train_frame = test_frame.assign(
            **{e: [5.0, 6.0, 5.5] for e in REGRESSION_ENDPOINTS},
            **{f"{e}_conf_low": [4.5, 5.5, 5.0] for e in REGRESSION_ENDPOINTS},
            **{f"{e}_conf_high": [5.5, 6.5, 6.0] for e in REGRESSION_ENDPOINTS},
        )

        class FakeModel:
            name = "fake"
            endpoints = tuple(REGRESSION_ENDPOINTS)

            def fit(self, observations):
                return self

            def predict(self, smiles):
                from cypshift.models.base import PredictiveDistribution

                values = np.full((1, len(smiles), len(REGRESSION_ENDPOINTS)), 5.0)
                return PredictiveDistribution(samples=values, endpoints=self.endpoints)

        monkeypatch.setattr(
            "cypshift.cli.load",
            lambda name, directory=None: test_frame if name == "test" else train_frame,
        )
        monkeypatch.setattr(
            "cypshift.cli.build_model", lambda config, surrogate=None, targets=None: FakeModel()
        )
        monkeypatch.setattr(
            "cypshift.cli.assemble_training_data", lambda config, frame: (pd.DataFrame(), None)
        )

        output = tmp_path / "submission.csv"
        result = CliRunner().invoke(app, ["submit", str(output)])
        assert result.exit_code == 0, result.output

        written = pd.read_csv(output)
        assert len(written) == len(test_frame)
        assert set(REGRESSION_ENDPOINTS) <= set(written.columns)
        assert written[list(REGRESSION_ENDPOINTS)].notna().all().all()

    def test_the_track_must_be_one_of_the_two(self):
        result = CliRunner().invoke(app, ["submit", "/tmp/unused.csv", "--track", "both"])
        assert result.exit_code != 0

    def test_the_track_option_is_registered(self):
        # Inspect the registered command rather than the rendered help: rich wraps the
        # help panel to the terminal width and its formatting differs across versions,
        # which made a substring check on --help flaky on CI.
        from typer.main import get_command

        submit = get_command(app).commands["submit"]
        options = [opt for param in submit.params for opt in param.opts]
        assert "--track" in options

    def test_scalar_models_can_also_predict_tdi_from_two_values(self):
        """The 'two values' path: scalar models predict the direct and preincubated
        pIC50 and compare them, so the classification track is not curve-only."""
        from cypshift.models.boosted import BoostedTrees
        from cypshift.models.neural import NeuralNetwork

        assert hasattr(BoostedTrees(), "predict_is_time_dependent")
        assert hasattr(NeuralNetwork(), "predict_is_time_dependent")
