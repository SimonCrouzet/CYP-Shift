"""Tracking must be optional and must not break a run when unavailable."""

from cypshift.experiment import ExperimentConfig
from cypshift.tracking import Tracker, track


def test_disabled_tracking_logs_without_a_backend():
    tracker = Tracker(ExperimentConfig()).start()
    tracker.log({"MA_ST-RAE": 0.68}, prefix="holdout")
    tracker.finish()


def test_the_context_manager_closes_even_on_failure():
    config = ExperimentConfig()
    try:
        with track(config) as tracker:
            tracker.log({"MA_ST-RAE": 0.68})
            raise RuntimeError("experiment failed")
    except RuntimeError:
        pass


def test_a_missing_backend_is_a_warning_not_an_error(monkeypatch):
    config = ExperimentConfig()
    config.override("tracking.enabled", True)
    monkeypatch.setitem(__import__("sys").modules, "wandb", None)
    tracker = Tracker(config).start()
    tracker.log({"MA_ST-RAE": 0.68})
    tracker.finish()


def test_configuration_is_logged_as_flat_dotted_keys():
    flat = ExperimentConfig().to_flat_dict()
    assert flat["conditioning.strategy"] == "factorised"
    assert flat["model.boosted.n_estimators"] == 600
