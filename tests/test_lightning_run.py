"""The Lightning AI sweep launcher: the orchestration logic that needs no cloud or SDK."""

import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lightning_run as lr  # noqa: E402 -- after the scripts path insert, as the other script tests do


def test_data_tar_sources_adds_external_only_when_requested():
    assert lr.data_tar_sources(False) == ["data/raw"]
    assert lr.data_tar_sources(True) == ["data/raw", "data/external"]


def test_sweep_command_carries_the_env_and_optional_setup():
    with_setup = lr.sweep_command("~/cyp", "python", workers=8, members=1, run_setup=True)
    assert "./setup.sh &&" in with_setup
    assert "CYP_WORKERS=8 CYP_MEMBERS=1" in with_setup
    assert with_setup.endswith("python scripts/night_run.py")
    assert "./setup.sh" not in lr.sweep_command("~/cyp", "python", 0, 0, run_setup=False)


def test_sweep_command_forwards_extra_env_after_the_sweep_vars():
    command = lr.sweep_command(
        "~/cyp", "python", 4, 1, run_setup=False, extra_env={"CYP_WANDB": "1", "WANDB_PROJECT": "p"}
    )
    assert "CYP_WANDB=1" in command and "WANDB_PROJECT=p" in command


def test_wandb_env_off_is_empty_and_on_forwards_present_vars(monkeypatch):
    assert lr.wandb_env(False) == {}
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    env = lr.wandb_env(True)
    assert (
        env["CYP_WANDB"] == "1" and env["WANDB_API_KEY"] == "k" and env["WANDB_PROJECT"] == "proj"
    )
    assert "WANDB_ENTITY" not in env  # an unset variable is not forwarded


def test_result_tar_command_guards_each_path_with_a_test():
    command = lr.result_tar_command("~/cyp", "results.tar.gz")
    # Every result path is added only if it exists, so a missing one does not fail the whole tar.
    for path in lr.RESULT_PATHS:
        assert f"test -e {path}" in command


def test_plan_is_pure_and_redacts_the_api_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "supersecret")
    args = lr.build_parser().parse_args(["--wandb", "--machine", "A100"])
    steps = lr.plan(args)
    blob = "\n".join(steps)
    assert "git archive HEAD" in blob and "night_run.py" in blob
    assert "A100" in blob
    assert "supersecret" not in blob and "WANDB_API_KEY=***" in blob  # key never printed


def test_resolve_machine_rejects_an_unknown_name():
    with pytest.raises(SystemExit, match="unknown --machine"):
        lr.resolve_machine("definitely-not-a-gpu")


def test_build_code_archive_ships_tracked_code_but_no_data(tmp_path):
    """git archive carries the committed tree (scripts/, src/) and not the gitignored data."""
    archive = lr.build_code_archive(tmp_path / "code.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert any(n.startswith("scripts/") for n in names)
    assert any(n.startswith("src/cypshift/") for n in names)
    assert not any(n.startswith("data/") for n in names)  # data is gitignored, never archived
