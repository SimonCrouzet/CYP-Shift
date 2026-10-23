"""The sweep's optional W&B monitoring: off by default, and never a gate on the run."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import night_run  # noqa: E402 -- after the scripts path insert, as the other script tests do


def test_wandb_start_is_a_no_op_without_the_opt_in(monkeypatch):
    """CYP_WANDB unset leaves no run, so an ordinary sweep has no W&B dependency or side effect."""
    monkeypatch.delenv("CYP_WANDB", raising=False)
    monkeypatch.setattr(night_run, "_WANDB_RUN", None)
    night_run.wandb_start()
    assert night_run._WANDB_RUN is None


def test_record_writes_the_csv_and_tolerates_wandb_being_off(monkeypatch, tmp_path):
    """record appends the result row and calls the W&B hook, which is a no-op when off."""
    results = tmp_path / "night_results.csv"
    monkeypatch.setattr(night_run, "RESULTS", results)
    monkeypatch.setattr(night_run, "_WANDB_RUN", None)
    row = {"rev": "abc", "phase": "reg", "name": "graph", "seed": 0, "status": "ok",
           "macro_ST_RAE": 0.78}  # fmt: skip
    night_run.record(row)
    written = pd.read_csv(results)
    assert written.loc[0, "name"] == "graph"
    assert written.loc[0, "macro_ST_RAE"] == 0.78


def test_wandb_log_row_is_a_no_op_when_no_run_is_active(monkeypatch):
    """With no active run the log hook returns without importing or calling wandb."""
    monkeypatch.setattr(night_run, "_WANDB_RUN", None)
    night_run.wandb_log_row({"phase": "reg", "name": "graph", "status": "ok", "macro_ST_RAE": 0.5})


def test_wandb_log_row_logs_a_table_and_scalars_to_an_active_run(monkeypatch):
    """With a run active, each row is logged as a rebuilt results table plus a phase-grouped
    scalar, so the dashboard shows a point per configuration as the sweep proceeds."""
    logged = []

    class FakeRun:
        def log(self, metrics):
            logged.append(metrics)

    monkeypatch.setattr(night_run, "_WANDB_RUN", FakeRun())
    monkeypatch.setattr(night_run, "_WANDB_ROWS", [])
    night_run.wandb_log_row({"phase": "reg", "name": "graph", "status": "ok", "macro_ST_RAE": 0.78})
    (metrics,) = logged
    assert metrics["configs_done"] == 1
    assert metrics["ST_RAE/reg/graph"] == 0.78
    assert "results" in metrics  # the running table is logged too


def test_wandb_start_survives_a_failed_init(monkeypatch):
    """A remote run with no usable key makes wandb.init raise; the sweep carries on without it."""
    import types

    def refuse(**_):
        raise RuntimeError("api_key not configured (no-tty)")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=refuse))
    monkeypatch.setenv("CYP_WANDB", "1")
    monkeypatch.setattr(night_run, "_WANDB_RUN", None)
    night_run.wandb_start()
    assert night_run._WANDB_RUN is None
