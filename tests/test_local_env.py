"""The gitignored per-machine token file: parsing, precedence, secrecy, and what the launcher
forwards from it."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from cypshift import local_env
from cypshift.local_env import load_local_env, parse_local_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lightning_run as lr  # noqa: E402 -- after the scripts path insert, as the other script tests do

REPO = Path(__file__).resolve().parent.parent


def test_parse_accepts_the_common_shapes() -> None:
    text = (
        "# a comment\n"
        "\n"
        "WANDB_API_KEY=abc123\n"
        "export HF_TOKEN = 'hf value'\n"
        'TABPFN_TOKEN="tab # not a comment"\n'
        "WANDB_PROJECT=proj # trailing comment\n"
    )
    assert parse_local_env(text) == {
        "WANDB_API_KEY": "abc123",
        "HF_TOKEN": "hf value",
        "TABPFN_TOKEN": "tab # not a comment",
        "WANDB_PROJECT": "proj",
    }


def test_blank_template_entries_are_skipped() -> None:
    assert parse_local_env("WANDB_API_KEY=\nHF_TOKEN=''\nLIGHTNING_API_KEY=   \n") == {}


def test_a_malformed_line_warns_by_number_and_never_echoes_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=local_env.logger.name):
        parsed = parse_local_env("GOOD=1\nsupersecretvalue\n9BAD=2\n")
    assert parsed == {"GOOD": "1"}
    assert "local.machine.env:2" in caplog.text and "local.machine.env:3" in caplog.text
    assert "supersecretvalue" not in caplog.text


def test_load_sets_only_what_is_unset_and_returns_names(tmp_path: Path) -> None:
    path = tmp_path / "local.machine.env"
    path.write_text("WANDB_API_KEY=from-file\nHF_TOKEN=hf-file\nTABPFN_TOKEN=\n")
    environ = {"WANDB_API_KEY": "exported"}
    assert load_local_env(path, environ) == ["HF_TOKEN"]
    assert environ == {"WANDB_API_KEY": "exported", "HF_TOKEN": "hf-file"}


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    assert load_local_env(tmp_path / "absent.env", environ) == []
    assert environ == {}


def test_the_path_can_be_redirected_outside_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "tokens.env"
    elsewhere.write_text("HF_TOKEN=elsewhere\n")
    monkeypatch.setenv(local_env.PATH_VARIABLE, str(elsewhere))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert local_env.default_path() == elsewhere
    assert load_local_env() == ["HF_TOKEN"]
    monkeypatch.delenv("HF_TOKEN")  # load wrote os.environ directly; do not leak it


def test_the_shipped_file_is_gitignored_and_loads_nothing() -> None:
    """The template exists for the user to fill in, must never be tracked, and ships blank."""
    shipped = REPO / local_env.LOCAL_ENV_FILE
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", local_env.LOCAL_ENV_FILE], cwd=REPO, check=False
    )
    assert ignored.returncode == 0
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", local_env.LOCAL_ENV_FILE],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    assert tracked.returncode != 0
    if shipped.exists():
        # every active line is NAME= or NAME=value; none is malformed
        for number, raw in enumerate(shipped.read_text().splitlines(), start=1):
            line = raw.strip()
            assert not line or line.startswith("#") or "=" in line, f"line {number}"


def _launch_args(**overrides: object) -> object:
    parser = lr.build_parser()
    args = parser.parse_args(["launch", "--dry-run", "--rev", "abc1234"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_sweep_tokens_are_forwarded_without_a_flag_and_only_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("TABPFN_TOKEN", "HF_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert lr.sweep_env(_launch_args(), "abc1234") == {"CYP_REV": "abc1234"}

    monkeypatch.setenv("TABPFN_TOKEN", "tab")
    monkeypatch.setenv("HF_TOKEN", "hf")
    env = lr.sweep_env(_launch_args(), "abc1234")
    assert env == {"CYP_REV": "abc1234", "TABPFN_TOKEN": "tab", "HF_TOKEN": "hf"}


def test_wandb_still_needs_its_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wb")
    assert "WANDB_API_KEY" not in lr.sweep_env(_launch_args(wandb=False), "abc1234")
    assert lr.sweep_env(_launch_args(wandb=True), "abc1234")["WANDB_API_KEY"] == "wb"


def test_the_dry_run_plan_redacts_every_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wb-secret")
    monkeypatch.setenv("TABPFN_TOKEN", "tab-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("WANDB_PROJECT", "visible-project")
    blob = "\n".join(lr.plan_launch(_launch_args(wandb=True)))
    for secret in ("wb-secret", "tab-secret", "hf-secret"):
        assert secret not in blob
    for name in ("WANDB_API_KEY", "TABPFN_TOKEN", "HF_TOKEN"):
        assert f"{name}='***'" in blob or f"{name}=***" in blob
    assert "visible-project" in blob  # only secrets are masked


def test_the_wandb_key_falls_back_to_what_wandb_login_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`wandb login` writes ~/.netrc and exports nothing; the launcher must still find the key."""
    netrc_file = tmp_path / ".netrc"
    netrc_file.write_text("machine api.wandb.ai\n  login user\n  password netrc-key\n")
    netrc_file.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert lr.wandb_api_key() == "netrc-key"
    assert lr.wandb_env(True)["WANDB_API_KEY"] == "netrc-key"

    monkeypatch.setenv("WANDB_API_KEY", "exported-key")  # an exported key wins
    assert lr.wandb_api_key() == "exported-key"


def test_no_wandb_key_anywhere_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))  # no .netrc here
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert lr.wandb_api_key() is None
    assert "WANDB_API_KEY" not in lr.wandb_env(True)


def test_launch_with_wandb_but_no_key_fails_before_any_cloud_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lr, "wandb_api_key", lambda: None)

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("the cloud must not be touched without a key")

    monkeypatch.setattr(lr, "open_studio", forbidden)
    with pytest.raises(SystemExit, match="wandb login"):
        lr.launch(_launch_args(wandb=True, dry_run=False))
