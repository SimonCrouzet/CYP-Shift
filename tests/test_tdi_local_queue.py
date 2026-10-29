"""The local TDI queue: which jobs exist, what they run, and that a finished job is skipped."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import night_run  # noqa: E402 -- after the scripts path insert
import tdi_local_queue as script  # noqa: E402


def test_the_stages_run_in_order_and_the_graph_comes_last():
    stages = [job.stage for job in script.jobs()]
    assert stages == sorted(stages, key=script.STAGES.index)
    assert stages[-1] == "graph" and set(stages) == set(script.STAGES)


def test_the_classifier_jobs_train_on_the_directly_measured_compounds_and_write_test_scores():
    job = next(j for j in script.jobs() if j.name == "lgb_chemeleon_covered")
    argv = list(job.argv)
    assert argv[argv.index("--train-on") + 1] == "covered"
    assert argv[argv.index("--test-output") + 1].endswith("f5_lgb_chemeleon_covered_test.csv")
    assert job.outputs == (Path(argv[argv.index("--test-output") + 1]),)


def test_the_tabpfn_timing_run_covers_one_fold_and_writes_no_test_scores():
    job = next(j for j in script.jobs() if j.stage == "tabpfn")
    argv = list(job.argv)
    assert argv[argv.index("--max-folds") + 1] == "1" and "--test-output" not in argv
    assert argv[argv.index("--pca") + 1] == "128"


def test_the_graph_jobs_run_one_seed_each_through_night_run():
    job = script.graph_job(2)
    assert job.argv == ("scripts/night_run.py",)
    assert job.env["CYP_SEEDS"] == "2" and job.env["CYP_CELLS"] == script.GRAPH_CELL
    assert job.env["CYP_JOBS"] == "1" and job.env["CYP_WORKERS"] == "0"
    assert job.outputs[0].parts[-2] == f"reg_{script.GRAPH_NAME}_seed2"


def test_the_graph_recipe_name_is_the_one_night_run_gives_the_cell():
    declared = {night_run._resolved_key(o): k for k, o in night_run.REGRESSION.items()}
    recipe = night_run._apply(
        dict(night_run.REGRESSION[night_run.FOCUS_BASE]), night_run._BOTH + night_run._BUDGET
    )
    cell = next(c for c in night_run.TDI_FOLD_CELLS if c.tag == script.GRAPH_CELL)
    overrides = night_run._apply(recipe, cell.sets)
    assert night_run._recipe_name(cell.tag, overrides, declared) == script.GRAPH_NAME


def test_the_novartis_job_runs_the_rate_cell_at_seed_zero_and_its_name_is_current():
    job = next(j for j in script.jobs() if j.stage == "novartis")
    assert job.env["CYP_CELLS"] == script.KOBS_CELL and job.env["CYP_SEEDS"] == "0"
    assert job.outputs[0].parts[-2] == f"reg_{script.KOBS_NAME}_seed0"
    declared = {night_run._resolved_key(o): k for k, o in night_run.REGRESSION.items()}
    recipe = night_run._apply(
        dict(night_run.REGRESSION[night_run.FOCUS_BASE]), night_run._BOTH + night_run._BUDGET
    )
    cell = next(c for c in night_run.TDI_FOLD_CELLS if c.tag == script.KOBS_CELL)
    overrides = night_run._apply(recipe, cell.sets)
    assert night_run._recipe_name(cell.tag, overrides, declared) == script.KOBS_NAME


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="stage"):
        script.selected("lightgbm,forest")


def test_a_job_with_its_files_is_skipped_and_the_others_run_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "ROOT", tmp_path)
    done = script.baseline_job("lgb_fp_dense_covered", "lightgbm")
    (tmp_path / done.outputs[0]).parent.mkdir(parents=True)
    (tmp_path / done.outputs[0]).write_text("scores")
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["tdi_local_queue.py", "--only", "lightgbm"])
    script.main()
    assert len(ran) == 2 and all(a[:3] == ["nice", "-n", "10"] for a in ran)
    assert not any("fp_dense" in " ".join(a) for a in ran)
    assert (tmp_path / script.LOGS / "lgb_chemeleon_covered.log").exists()


def test_a_failed_job_is_reported_and_the_queue_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "ROOT", tmp_path)
    ran = []

    def fake_run(argv, **kwargs):
        ran.append(argv)
        return subprocess.CompletedProcess(argv, 1 if len(ran) == 1 else 0)

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["tdi_local_queue.py", "--only", "lightgbm"])
    with pytest.raises(SystemExit):
        script.main()
    assert len(ran) == 3
