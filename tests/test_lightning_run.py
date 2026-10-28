"""The Lightning AI sweep launcher, run against a fake local Studio (no cloud, no SDK)."""

import io
import shutil
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lightning_run as lr  # noqa: E402 -- after the scripts path insert, as the other script tests do
import night_run  # noqa: E402

HEADER = "rev,phase,name,seed,status,macro_ST_RAE"


class FakeStudio:
    """A Studio whose filesystem is a local directory and whose ``run`` executes in it.

    Mirrors the SDK surface the launcher uses (``status``/``start``/``stop``/``run``/
    ``run_with_exit_code``/``upload_file``/``download_file``) including its sharp edges: ``start``
    raises on a running Studio and ``run`` raises on a non-zero exit.
    """

    def __init__(self, root: Path, running: bool = False) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._running = running
        self.started_with = None

    @property
    def status(self):
        return types.SimpleNamespace(name="Running" if self._running else "Stopped")

    def start(self, machine, max_runtime=None) -> None:
        if self._running:
            raise RuntimeError("Studio is already running")
        self._running = True
        self.started_with = (machine, max_runtime)

    def stop(self) -> None:
        self._running = False

    def _sh(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            shell=True,
            cwd=self.root,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )

    def run(self, *commands: str) -> str:
        out = []
        for command in commands:
            result = self._sh(command)
            if result.returncode != 0:
                raise RuntimeError(f"exit {result.returncode}: {result.stderr}")
            out.append(result.stdout)
        return "".join(out)

    def run_with_exit_code(self, *commands: str):
        result = self._sh(commands[0])
        return result.stdout, result.returncode

    def upload_file(self, file_path, remote_path=None, progress_bar=True) -> None:
        destination = self.root / remote_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, destination)

    def download_file(self, remote_path, file_path=None) -> None:
        shutil.copy(self.root / remote_path, file_path)


def _release_file(tmp_path: Path) -> Path:
    return tmp_path / "release_sweep"


def _stub_sweep(tmp_path: Path) -> Path:
    """A stand-in for night_run.py: records one result row under CYP_REV, then stays alive until
    the test creates the release file (capped at 60 s so a failed test cannot orphan it). Blocking
    on a file rather than sleeping keeps "the sweep is running" deterministic under load.
    Invoked as ``<stub> scripts/night_run.py``."""
    stub = tmp_path / "stub_sweep.sh"
    if stub.exists():
        # Never rewrite it: a launch that is already running executes this file, and a shell reads
        # its script incrementally, so truncating it under a live sweep can end the sweep early.
        return stub
    stub.write_text(
        "#!/bin/sh\n"
        "mkdir -p data\n"
        f"printf '{HEADER}\\n%s,reg,graph,0,ok,0.78\\n' \"$CYP_REV\" > data/night_results.csv\n"
        "i=0\n"
        f"while [ ! -f '{_release_file(tmp_path)}' ] && [ $i -lt 600 ]; "
        "do sleep 0.1; i=$((i+1)); done\n"
    )
    stub.chmod(0o755)
    return stub


def _let_the_sweep_finish(tmp_path: Path) -> None:
    _release_file(tmp_path).touch()


def _fake_code_archive(path: Path, ref: str = "HEAD") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        payload = b"# stand-in for the committed tree\n"
        info = tarfile.TarInfo("scripts/night_run.py")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A local 'repo' with core data and one pre-existing result row the merge must preserve."""
    repo = tmp_path / "local"
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "data" / "raw" / "core.csv").write_text("a\n1\n")
    (repo / "data" / "night_results.csv").write_text(f"{HEADER}\nold,reg,local_cfg,0,ok,0.9\n")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(lr, "build_code_archive", _fake_code_archive)
    return repo


def _wait_until_stopped(studio: FakeStudio, timeout: float = 15.0) -> dict:
    """Wait for the (already released) stub sweep to exit."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = lr.read_status(studio, lr.DEFAULT_REMOTE_DIR)
        if not current["running"]:
            return current
        time.sleep(0.2)
    raise AssertionError("the stub sweep never finished")


def _launch_args(tmp_path: Path, *extra: str):
    return lr.build_parser().parse_args(
        [
            "launch",
            "--no-setup",
            "--rev",
            "abc1234",
            "--remote-python",
            str(_stub_sweep(tmp_path)),
            *extra,
        ]
    )


def test_launch_runs_detached_then_fetch_merges_without_clobbering(tmp_path, local):
    """The persistent flow end to end: start the Studio, sync code and data, run the sweep in the
    background under the archived revision, observe it via status, and fetch it without losing the
    local results already in the repo."""
    studio = FakeStudio(tmp_path / "studio")
    lr.launch(_launch_args(tmp_path), studio=studio, resolver=lambda name: name)

    assert studio.started_with == ("L4", None)  # the default machine is a real SDK name, not A10G
    remote = studio.root / lr.DEFAULT_REMOTE_DIR
    assert (remote / "scripts" / "night_run.py").exists()  # code synced
    assert (remote / "data" / "raw" / "core.csv").exists()  # data synced
    assert (remote / lr.DATA_MARKER).exists()
    assert lr.read_status(studio, lr.DEFAULT_REMOTE_DIR)["running"]  # started detached, still going

    _let_the_sweep_finish(tmp_path)
    finished = _wait_until_stopped(studio)
    assert finished["rows"] == 1 and finished["project"]

    lr.fetch(lr.build_parser().parse_args(["fetch"]), studio)
    merged = pd.read_csv(local / "data" / "night_results.csv")
    assert set(merged["name"]) == {"local_cfg", "graph"}  # the local row survived the merge
    assert "abc1234" in set(merged["rev"])  # the sweep ran under the archived revision (CYP_REV)


def test_a_second_launch_is_refused_while_the_sweep_runs(tmp_path, local):
    studio = FakeStudio(tmp_path / "studio")
    lr.launch(_launch_args(tmp_path), studio=studio, resolver=lambda name: name)
    with pytest.raises(SystemExit, match="already running"):
        lr.launch(_launch_args(tmp_path), studio=studio, resolver=lambda name: name)
    _let_the_sweep_finish(tmp_path)
    _wait_until_stopped(studio)


def test_a_running_studio_is_reused_not_restarted(tmp_path, local):
    """The SDK's start() raises on a running Studio; a persistent Studio is usually already up."""
    studio = FakeStudio(tmp_path / "studio", running=True)
    lr.launch(_launch_args(tmp_path), studio=studio, resolver=lambda name: name)
    assert studio.started_with is None  # never called start()
    _let_the_sweep_finish(tmp_path)
    _wait_until_stopped(studio)


def test_data_is_uploaded_once_and_skipped_afterwards(tmp_path, local):
    studio = FakeStudio(tmp_path / "studio", running=True)
    args = _launch_args(tmp_path)
    assert lr.sync_data(studio, args, Path(args.data_archive)) is True
    assert lr.sync_data(studio, args, Path(args.data_archive)) is False  # persistent: already there
    refreshed = _launch_args(tmp_path, "--refresh-data")
    assert lr.sync_data(studio, refreshed, Path(refreshed.data_archive)) is True


def test_adding_external_data_after_a_raw_only_sync_ships_the_external_caches(tmp_path, local):
    """The marker records what was shipped. A raw-only sync followed by ``--data-external`` used to
    skip on the marker's mere presence and leave the Studio with the smaller panel."""
    (local / "data" / "external").mkdir()
    (local / "data" / "external" / "aux.csv").write_text("c\n3\n")
    studio = FakeStudio(tmp_path / "studio", running=True)
    raw_only = _launch_args(tmp_path)
    assert lr.sync_data(studio, raw_only, Path(raw_only.data_archive)) is True
    remote = studio.root / lr.DEFAULT_REMOTE_DIR
    assert not (remote / "data" / "external").exists()

    both = _launch_args(tmp_path, "--data-external")
    assert lr.sync_data(studio, both, Path(both.data_archive)) is True
    assert (remote / "data" / "external" / "aux.csv").read_text() == "c\n3\n"
    assert lr.marker_sources((remote / lr.DATA_MARKER).read_text()) == {
        "data/raw",
        "data/external",
    }
    assert lr.sync_data(studio, both, Path(both.data_archive)) is False  # now complete
    assert lr.sync_data(studio, raw_only, Path(raw_only.data_archive)) is False  # a subset


def test_a_legacy_marker_counts_as_raw_only(tmp_path, local):
    """Earlier launchers wrote an empty / ``ok`` marker that did not say what was shipped; it is
    read as the core data only, so the external caches are re-sent rather than assumed present."""
    (local / "data" / "external").mkdir()
    (local / "data" / "external" / "aux.csv").write_text("c\n3\n")
    studio = FakeStudio(tmp_path / "studio", running=True)
    remote = studio.root / lr.DEFAULT_REMOTE_DIR
    (remote / "data").mkdir(parents=True)
    (remote / lr.DATA_MARKER).write_text("")
    args = _launch_args(tmp_path, "--data-external")
    assert lr.sync_data(studio, args, Path(args.data_archive)) is True
    assert (remote / "data" / "external" / "aux.csv").exists()


def test_stop_refuses_a_running_sweep_unless_forced(tmp_path, local):
    studio = FakeStudio(tmp_path / "studio")
    lr.launch(_launch_args(tmp_path), studio=studio, resolver=lambda name: name)
    stop_args = lr.build_parser().parse_args(["stop"])
    with pytest.raises(SystemExit, match="still running"):
        lr.stop(stop_args, studio)
    _let_the_sweep_finish(tmp_path)
    _wait_until_stopped(studio)
    lr.stop(stop_args, studio)
    assert not lr.studio_is_running(studio)


def test_parse_status_reads_the_snippet_output():
    parsed = lr.parse_status("PROJECT=present\nSWEEP=running\nROWS=7\n---\nline a\nline b\n")
    assert parsed == {"project": True, "running": True, "rows": 7, "tail": "line a\nline b"}
    assert lr.parse_status("PROJECT=missing\n")["project"] is False


def test_merge_results_adds_only_new_rows(tmp_path):
    local_csv, incoming = tmp_path / "local.csv", tmp_path / "incoming.csv"
    local_csv.write_text(f"{HEADER}\nr1,reg,a,0,ok,0.8\n")
    incoming.write_text(f"{HEADER}\nr1,reg,a,0,ok,0.8\nr1,reg,b,0,ok,0.7\n")
    assert lr.merge_results(local_csv, incoming) == 1  # the duplicate is not re-added
    assert lr.merge_results(local_csv, incoming) == 0  # fetching again mid-run is a no-op
    assert len(pd.read_csv(local_csv)) == 2


def test_extract_results_places_bundles_and_merges_the_csv(tmp_path):
    archive = tmp_path / "results.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in (
            ("data/night_results.csv", f"{HEADER}\nr1,reg,a,0,ok,0.8\n"),
            ("data/run_bundles/r1/reg_a_seed0/oof.csv", "x\n1\n"),
        ):
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    repo = tmp_path / "repo"
    assert lr.extract_results(archive, repo) == 1
    assert (repo / "data" / "run_bundles" / "r1" / "reg_a_seed0" / "oof.csv").exists()
    assert not (repo / "scratch" / "lightning_extract").exists()  # staging cleaned up


def test_detached_command_survives_the_session_and_records_its_pid():
    command = lr.detached_sweep_command(
        "cyp-shift", "./.env/bin/python", 8, 1, {"CYP_REV": "abc1234", "WANDB_PROJECT": "my proj"}
    )
    assert "nohup env" in command and "< /dev/null" in command  # outlives the connection
    assert "CYP_WORKERS=8 CYP_MEMBERS=1 CYP_REV=abc1234" in command
    assert "WANDB_PROJECT='my proj'" in command  # values are shell-quoted
    assert f"echo $! > {lr.PID_FILE}" in command and f"> {lr.LOG_FILE}" in command


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
    command = lr.result_tar_command("cyp-shift", "results.tar.gz")
    for path in lr.RESULT_PATHS:  # a missing path must not fail the whole tar
        assert f"test -e {path}" in command


def test_the_plan_is_pure_redacts_the_key_and_uses_a_relative_remote_dir(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "supersecret")
    args = lr.build_parser().parse_args(
        ["launch", "--wandb", "--machine", "H100", "--rev", "abc1234"]
    )
    blob = "\n".join(lr.plan_launch(args))
    assert "H100" in blob and "night_run.py" in blob and "CYP_REV=abc1234" in blob
    assert "supersecret" not in blob and "WANDB_API_KEY=" in blob and "***" in blob
    assert "~" not in lr.DEFAULT_REMOTE_DIR  # the file API does not expand a shell ~


def test_resolve_machine_validates_against_the_sdk(monkeypatch):
    fake_sdk = types.ModuleType("lightning_sdk")
    fake_sdk.Machine = types.SimpleNamespace(L4="<L4>", T4="<T4>")
    fake_sdk.Studio = object
    monkeypatch.setitem(sys.modules, "lightning_sdk", fake_sdk)
    assert lr.resolve_machine("l4") == "<L4>"
    with pytest.raises(SystemExit, match="A10G"):  # the old default is not an SDK machine
        lr.resolve_machine("A10G")


def test_the_sweep_revision_can_be_overridden_for_an_archive_without_git(monkeypatch):
    """A git archive has no .git, so the launcher supplies CYP_REV; otherwise every code version
    would share the 'nogit' results namespace."""
    monkeypatch.setenv("CYP_REV", "abc1234")
    assert night_run.run_identity() == "abc1234"


def test_build_code_archive_ships_tracked_code_but_no_data(tmp_path):
    """git archive carries the committed tree (scripts/, src/) and not the gitignored data."""
    archive = lr.build_code_archive(tmp_path / "code.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert any(n.startswith("scripts/") for n in names)
    assert any(n.startswith("src/cypshift/") for n in names)
    assert not any(n.startswith("data/") for n in names)  # data is gitignored, never archived


def test_an_unresolved_teamspace_becomes_guidance_before_any_studio_exists(monkeypatch):
    """The SDK raises a bare ValueError for an unresolved teamspace; we say what to set."""

    class NoTeamspace:
        def __init__(self, **_):
            raise ValueError("Couldn't resolve teamspace from the provided name, org, or user")

    monkeypatch.setattr(lr, "_require_sdk", lambda: (NoTeamspace, object))
    monkeypatch.setattr(
        lr, "teamspace_hint", lambda: " Logged in as me; teamspaces: ts (owner: me)."
    )
    args = lr.build_parser().parse_args(["launch", "--dry-run"])
    with pytest.raises(SystemExit) as raised:
        lr.open_studio(args)
    message = str(raised.value)
    assert "--teamspace" in message and "LIGHTNING_TEAMSPACE" in message
    assert "Logged in as me; teamspaces: ts (owner: me)." in message


def test_the_teamspace_owner_flags_reach_the_sdk(monkeypatch):
    seen = {}

    class Recording:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(lr, "_require_sdk", lambda: (Recording, object))
    args = lr.build_parser().parse_args(
        ["launch", "--dry-run", "--teamspace", "ts", "--user", "me", "--studio", "s1"]
    )
    lr.open_studio(args)
    assert seen == {"name": "s1", "teamspace": "ts", "user": "me", "org": None, "create_ok": True}


def test_other_value_errors_are_not_rewritten(monkeypatch):
    class Broken:
        def __init__(self, **_):
            raise ValueError("something else entirely")

    monkeypatch.setattr(lr, "_require_sdk", lambda: (Broken, object))
    with pytest.raises(ValueError, match="something else"):
        lr.open_studio(lr.build_parser().parse_args(["launch", "--dry-run"]))


def test_the_teamspace_hint_never_raises(monkeypatch):
    """With the SDK unavailable the hint says so, rather than raising or going silent."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("lightning_sdk"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    hint = lr.teamspace_hint()
    assert "could not look up your account" in hint and "ImportError" in hint


def _install_fake_account(monkeypatch, owners, fail_lookup=None):
    """A fake ``lightning_sdk.utils.resolve`` whose authed user owns ``owners`` (name -> names)."""
    import types

    class Owner:
        def __init__(self, name, teamspaces):
            self.name = name
            self._teamspaces = teamspaces

        @property
        def teamspaces(self):
            if isinstance(self._teamspaces, Exception):
                raise self._teamspaces
            return [types.SimpleNamespace(name=n) for n in self._teamspaces]

    def authed():
        if fail_lookup:
            raise fail_lookup
        user = Owner("me", owners["me"])
        user.organizations = [Owner(n, t) for n, t in owners.items() if n != "me"]
        return user

    module = types.ModuleType("lightning_sdk.utils.resolve")
    module._get_authed_user = authed
    monkeypatch.setitem(sys.modules, "lightning_sdk", types.ModuleType("lightning_sdk"))
    monkeypatch.setitem(sys.modules, "lightning_sdk.utils", types.ModuleType("lightning_sdk.utils"))
    monkeypatch.setitem(sys.modules, "lightning_sdk.utils.resolve", module)


def test_the_teamspace_hint_names_user_and_org_teamspaces(monkeypatch):
    _install_fake_account(monkeypatch, {"me": [], "my-org": ["main", "scratch"]})
    hint = lr.teamspace_hint()
    assert "user 'me': (none)" in hint
    assert "org 'my-org': 'main', 'scratch'" in hint


def test_the_teamspace_hint_reports_why_it_failed_not_silence(monkeypatch):
    class BlockedError(Exception):
        status = 403

    _install_fake_account(
        monkeypatch, {"me": []}, fail_lookup=BlockedError("Forbidden\n<html>" + "x" * 500)
    )
    hint = lr.teamspace_hint()
    assert "BlockedError 403: Forbidden" in hint
    assert "<html>" not in hint  # only the first line, never a page of HTML


def test_one_unreadable_owner_does_not_hide_the_others(monkeypatch):
    _install_fake_account(monkeypatch, {"me": ["solo"], "my-org": RuntimeError("no access")})
    hint = lr.teamspace_hint()
    assert "user 'me': 'solo'" in hint
    assert "org 'my-org': unreadable (RuntimeError: no access)" in hint
