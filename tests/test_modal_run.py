"""The Modal sweep launcher, run against a fake in-memory Modal (no cloud, no SDK needed)."""

import inspect
import shutil
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import lightning_run  # noqa: E402
import modal_run as mr  # noqa: E402 -- after the scripts path insert, as the other script tests do

REPO = Path(__file__).resolve().parent.parent


class NotFoundError(Exception):
    """Stands in for ``modal.exception.NotFoundError``."""


class FakeVolume:
    """An in-memory Volume with the client-side surface the launcher uses."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploads = 0

    def _key(self, path: str) -> str:
        return path.strip("/")

    def hydrate(self) -> "FakeVolume":
        return self

    def listdir(self, path: str) -> list[str]:
        key = self._key(path)
        hits = [f for f in self.files if f == key or f.startswith(key + "/")]
        if not hits:
            # modal 1.6.2: listdir / iterdir raise the SDK's NotFoundError, not FileNotFoundError.
            raise NotFoundError("No such file or directory")
        return hits

    def iterdir(self, path: str, recursive: bool = True):
        for found in self.listdir(path):
            yield types.SimpleNamespace(path=found, type=types.SimpleNamespace(name="FILE"))

    def read_file(self, path: str):
        key = self._key(path)
        if key not in self.files:
            raise FileNotFoundError(path)  # read_file does raise the builtin
        yield self.files[key]

    @contextmanager
    def batch_upload(self, force: bool = False):
        volume = self

        class Batch:
            def put_directory(self, local: str, remote: str) -> None:
                for file in Path(local).rglob("*"):
                    if file.is_file():
                        volume.files[volume._key(remote) + "/" + str(file.relative_to(local))] = (
                            file.read_bytes()
                        )

            def put_file(self, source: object, remote: str) -> None:
                data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
                volume.files[volume._key(remote)] = data

        volume.uploads += 1
        yield Batch()


class MissingVolume:
    """What ``Volume.from_name`` returns for a Volume that was never created."""

    def __init__(self, name: str) -> None:
        self.name = name

    def hydrate(self) -> None:
        raise NotFoundError(f"Volume {self.name!r} not found in environment 'main'")


class FakeSandbox:
    def __init__(self, args: tuple, kwargs: dict) -> None:
        self.args, self.kwargs = args, kwargs
        self.object_id = "sb-fake"
        self.running = True
        self.log: str | None = None  # what `tail` prints inside the Sandbox; None: exec fails

    def exec(self, *argv: str, timeout: int | None = None):
        if self.log is None:
            raise ConnectionError("exec unavailable")
        out = types.SimpleNamespace(read=lambda: self.log)
        return types.SimpleNamespace(stdout=out, wait=lambda: 0)

    def get_dashboard_url(self) -> str:
        return "https://modal.example/sb-fake"

    def terminate(self, wait: bool = False) -> None:
        self.running = False


class FakeImage:
    """Records the build chain; the launcher only ever chains."""

    def __init__(self, ops: tuple = ()) -> None:
        self.ops = ops

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: FakeImage((*self.ops, (name, args, kwargs)))


def make_fake_modal() -> types.SimpleNamespace:
    volumes: dict[str, FakeVolume] = {}
    sandboxes: list[FakeSandbox] = []

    def create(*args, **kwargs):
        sandbox = FakeSandbox(args, kwargs)
        sandboxes.append(sandbox)
        return sandbox

    def from_name(app_name, name, **_):
        live = [s for s in sandboxes if s.running]
        if not live:
            raise NotFoundError(name)
        return live[-1]

    def volume_from_name(name, create_if_missing=False, **_):
        if create_if_missing:
            return volumes.setdefault(name, FakeVolume())
        return volumes.get(name) or MissingVolume(name)

    return types.SimpleNamespace(
        App=types.SimpleNamespace(lookup=lambda name, create_if_missing=False: f"app:{name}"),
        Volume=types.SimpleNamespace(from_name=volume_from_name),
        Sandbox=types.SimpleNamespace(create=create, from_name=from_name),
        Secret=types.SimpleNamespace(from_dict=lambda env: types.SimpleNamespace(env=dict(env))),
        Image=types.SimpleNamespace(debian_slim=lambda python_version=None: FakeImage()),
        exception=types.SimpleNamespace(NotFoundError=NotFoundError),
        _volumes=volumes,
        _sandboxes=sandboxes,
    )


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A working directory with tiny data, a staged-code stub, and no tokens in the environment."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/train.csv").write_text("a,b\n1,2\n")
    (tmp_path / "data/external").mkdir()
    (tmp_path / "data/external/aux.csv").write_text("c\n3\n")
    for name in ("WANDB_API_KEY", "TABPFN_TOKEN", "HF_TOKEN", "WANDB_PROJECT", "WANDB_ENTITY"):
        monkeypatch.delenv(name, raising=False)
    # Neither this machine's `wandb login` nor its `hf auth login` may leak into the assertions.
    monkeypatch.setattr(mr, "wandb_api_key", lambda: None)
    monkeypatch.setattr(lightning_run, "wandb_api_key", lambda: None)
    monkeypatch.setattr(lightning_run, "hf_token", lambda: None)
    monkeypatch.setattr(mr, "git_revision", lambda ref="HEAD": "abc1234")

    def fake_stage(archive: Path, stage: Path, ref: str) -> Path:
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "pyproject.toml", stage / "pyproject.toml")
        return stage

    monkeypatch.setattr(mr, "stage_code", fake_stage)
    return tmp_path


def parse(*argv: str):
    return mr.build_parser().parse_args(["launch", *argv])


# --- pure pieces ---------------------------------------------------------------------------------


def test_image_requirements_follow_pyproject_and_include_the_run_extras() -> None:
    requirements = mr.image_requirements(REPO / "pyproject.toml")
    joined = " ".join(requirements)
    for needed in ("rdkit", "lightgbm", "torch", "chemprop", "tabpfn", "wandb"):
        assert needed in joined
    assert "pytest" not in joined and "lightning-sdk" not in joined  # dev/cloud extras stay out
    assert len(requirements) == len(set(requirements))


def test_the_entry_script_parses() -> None:
    parsed = subprocess.run(
        ["bash", "-n", "-c", mr.sweep_script()], capture_output=True, text=True, check=False
    )
    assert parsed.returncode == 0, parsed.stderr


def _run_script(
    tmp_path: Path, night_run_body: str
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Execute the real entry script against temp dirs with a stub interpreter."""
    mount, work = tmp_path / "vol", tmp_path / "work"
    (work / "scripts").mkdir(parents=True)
    (work / "configs").mkdir()
    stub = tmp_path / "py"
    stub.write_text(
        f"""#!/bin/bash
if [ "$1" = "-c" ]; then echo "imports ok; cuda available: stub"; exit 0; fi
{night_run_body}
"""
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", mr.sweep_script(str(mount), str(work), str(stub))],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, mount, work


def test_the_entry_script_links_outputs_into_the_volume_and_records_success(tmp_path) -> None:
    body = (
        "mkdir -p data/run_bundles submissions; "
        "echo 'rev,name' > data/night_results.csv; echo x > data/run_bundles/b.json; "
        "echo '{}' > configs/night_best.json; echo s > submissions/sub.csv; exit 0"
    )
    result, mount, work = _run_script(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert (mount / "data/night_results.csv").read_text() == "rev,name\n"
    assert (mount / "data/run_bundles/b.json").exists()
    assert (mount / "configs/night_best.json").read_text() == "{}\n"
    assert (mount / "submissions/sub.csv").exists()
    assert (mount / "sweep.exit").read_text().strip() == "0"
    assert "imports ok" in (mount / "sweep.log").read_text()
    assert (work / "data").is_symlink() and (work / "submissions").is_symlink()


def test_the_entry_script_records_a_failing_exit_and_clears_a_stale_one(tmp_path) -> None:
    mount = tmp_path / "vol"
    mount.mkdir()
    (mount / "sweep.exit").write_text("0\n")  # left by an earlier, completed run
    result, mount, _ = _run_script(tmp_path, "exit 3")
    assert result.returncode == 0
    assert (mount / "sweep.exit").read_text().strip() == "3"


def test_the_entry_script_never_deletes_a_non_empty_directory(tmp_path) -> None:
    mount, work = tmp_path / "vol", tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "data/precious.csv").write_text("keep")
    stub = tmp_path / "py"
    stub.write_text("#!/bin/bash\nexit 0\n")
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", mr.sweep_script(str(mount), str(work), str(stub))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "refusing" in result.stdout
    assert (work / "data/precious.csv").read_text() == "keep"


def test_status_outcomes() -> None:
    assert mr.parse_status(True, None, "h\nr1\n", "")["outcome"] == "running"
    assert mr.parse_status(False, "0\n", "h\nr1\nr2\n", "")["outcome"] == "completed"
    assert mr.parse_status(False, "1\n", None, "")["outcome"] == "failed"
    stopped = mr.parse_status(False, None, "h\nr1\n", "tail")
    assert stopped["outcome"] == "stopped" and stopped["rows"] == 1 and stopped["tail"] == "tail"
    assert mr.parse_status(False, None, None, "")["rows"] == 0


def test_secrets_are_kept_out_of_the_plain_environment(local, monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", "tab-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wb-secret")
    monkeypatch.setattr(mr, "wandb_api_key", lambda: "wb-secret")
    monkeypatch.setattr(lightning_run, "wandb_api_key", lambda: "wb-secret")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    plain, secret = mr.sandbox_environment(parse("--wandb"), "abc1234")
    assert secret == {"WANDB_API_KEY": "wb-secret", "TABPFN_TOKEN": "tab-secret"}
    assert plain["CYP_WANDB"] == "1" and plain["WANDB_PROJECT"] == "proj"
    assert plain["CYP_REV"] == "abc1234" and plain["HF_HOME"] == "/vol/hf_cache"
    assert "secret" not in " ".join(plain.values())


def test_the_dry_run_plan_masks_secrets_and_touches_no_sdk(local, monkeypatch, capsys) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", "tab-secret")
    monkeypatch.setattr(mr, "_require_modal", lambda: pytest.fail("dry run must not import modal"))
    monkeypatch.chdir(REPO)
    mr.main(["launch", "--dry-run", "--data-external"])
    out = capsys.readouterr().out
    assert "tab-secret" not in out and "TABPFN_TOKEN=***" in out
    assert "Sandbox.create" in out and "data/external" in out


# --- launch / status / fetch / stop against the fake Modal ----------------------------------------


def test_launch_uploads_the_data_once_and_starts_a_detached_sandbox(local, monkeypatch) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", "tab-secret")
    modal = make_fake_modal()
    sandbox = mr.launch(parse("--data-external", "--gpu", "A10", "--max-hours", "30"), modal)

    volume = modal._volumes["cyp-shift-sweep"]
    assert volume.files["data/raw/train.csv"] == b"a,b\n1,2\n"
    assert volume.files["data/external/aux.csv"] == b"c\n3\n"
    assert "data/.cyp_synced" in volume.files

    kwargs = sandbox.kwargs
    assert sandbox.args[:2] == ("bash", "-c")
    assert kwargs["gpu"] == "A10" and kwargs["name"] == "sweep" and kwargs["workdir"] == mr.WORKDIR
    assert kwargs["timeout"] == 24 * 3600  # capped
    assert kwargs["volumes"] == {mr.MOUNT: volume}
    assert kwargs["env"]["CYP_REV"] == "abc1234"
    assert [s.env for s in kwargs["secrets"]] == [{"TABPFN_TOKEN": "tab-secret"}]
    assert "tab-secret" not in " ".join(kwargs["env"].values())


def test_a_second_launch_does_not_resend_the_data(local) -> None:
    modal = make_fake_modal()
    mr.launch(parse(), modal)
    modal._sandboxes[0].running = False  # the first sweep ended
    mr.launch(parse(), modal)
    volume = modal._volumes["cyp-shift-sweep"]
    assert volume.uploads == 1 and len(modal._sandboxes) == 2

    modal._sandboxes[1].running = False
    mr.launch(parse("--refresh-data"), modal)
    assert volume.uploads == 2


def test_launch_refuses_while_a_sweep_is_running(local) -> None:
    modal = make_fake_modal()
    mr.launch(parse(), modal)
    with pytest.raises(SystemExit, match="already running"):
        mr.launch(parse(), modal)
    assert len(modal._sandboxes) == 1


def test_launch_with_wandb_and_no_key_fails_before_touching_modal(local) -> None:
    def forbidden(*_):
        raise AssertionError("must not reach Modal")

    modal = make_fake_modal()
    modal.App.lookup = forbidden
    with pytest.raises(SystemExit, match="wandb login"):
        mr.launch(parse("--wandb"), modal)


def test_status_reads_the_volume_and_the_sandbox(local, capsys) -> None:
    modal = make_fake_modal()
    mr.launch(parse(), modal)
    volume = modal._volumes["cyp-shift-sweep"]
    volume.files["data/night_results.csv"] = b"rev,name\nabc,g1\nabc,g2\n"
    volume.files["sweep.log"] = b"one\ntwo\nthree\n"
    running = mr.status(parse(), modal)
    assert running["outcome"] == "running" and running["rows"] == 2
    assert "three" in capsys.readouterr().out  # exec fails: falls back to the Volume copy

    modal._sandboxes[0].log = "live tail\n"  # the Volume copy of an open log lags the Sandbox's
    mr.status(parse(), modal)
    out = capsys.readouterr().out
    assert "live tail" in out and "three" not in out

    modal._sandboxes[0].running = False
    volume.files["sweep.exit"] = b"0\n"
    assert mr.status(parse(), modal)["outcome"] == "completed"


def test_fetch_merges_the_results_csv_and_copies_the_rest(local) -> None:
    modal = make_fake_modal()
    volume = modal.Volume.from_name("cyp-shift-sweep", create_if_missing=True)
    volume.files["data/night_results.csv"] = b"rev,name\nabc,remote\n"
    volume.files["data/run_bundles/abc/b.json"] = b"{}"
    volume.files["configs/night_best.json"] = b'{"k": 1}'
    volume.files["submissions/night_submission_regression.csv"] = b"id\n1\n"
    (local / "data/night_results.csv").write_text("rev,name\nold,local\n")

    added = mr.fetch(parse(), modal)

    assert added == 1
    merged = (local / "data/night_results.csv").read_text()
    assert "old,local" in merged and "abc,remote" in merged  # local rows survive
    assert (local / "data/run_bundles/abc/b.json").read_text() == "{}"
    assert (local / "configs/night_best.json").read_text() == '{"k": 1}'
    assert (local / "submissions/night_submission_regression.csv").exists()
    assert not (local / "scratch/modal_extract").exists()


def test_fetch_with_nothing_on_the_volume_is_not_an_error(local) -> None:
    modal = make_fake_modal()
    modal.Volume.from_name("cyp-shift-sweep", create_if_missing=True)
    assert mr.fetch(parse(), modal) == 0


def test_status_and_fetch_before_any_launch_say_to_launch_first(local) -> None:
    """A missing Volume raises the same NotFoundError as a missing path; it is ruled out first."""
    modal = make_fake_modal()
    for command in (mr.status, mr.fetch):
        with pytest.raises(SystemExit, match="run `launch` first"):
            command(parse(), modal)


def test_a_missing_path_is_absent_whichever_error_the_sdk_raises() -> None:
    assert mr._is_missing_path(NotFoundError("No such file or directory"))
    assert mr._is_missing_path(FileNotFoundError("/sweep.exit"))
    assert not mr._is_missing_path(ConnectionError("blip"))
    assert mr.volume_has(FakeVolume(), mr.DATA_MARKER) is False


def test_stop_terminates_only_a_running_sandbox(local, capsys) -> None:
    modal = make_fake_modal()
    mr.stop(parse(), modal)
    assert "no sweep Sandbox" in capsys.readouterr().out
    mr.launch(parse(), modal)
    mr.stop(parse(), modal)
    assert modal._sandboxes[0].running is False


def test_a_flaky_listing_does_not_trigger_a_data_reupload(local) -> None:
    """Only a clean not-found means 'absent'; any other failure must propagate."""

    class Flaky(FakeVolume):
        def listdir(self, path: str):
            raise ConnectionError("network blip")

    with pytest.raises(ConnectionError):
        mr.volume_has(Flaky(), mr.DATA_MARKER)


# --- contract with the real SDK (skipped until `modal` is installed) -----------------------------


def test_the_sandbox_call_matches_the_real_sdk_signature(local, monkeypatch) -> None:
    modal = pytest.importorskip("modal")
    monkeypatch.setenv("TABPFN_TOKEN", "t")
    plain, _ = mr.sandbox_environment(parse(), "abc1234")
    options = mr.sandbox_options(parse("--runtime", "gvisor"), plain)
    signature = inspect.signature(modal.Sandbox.create)
    signature.bind(
        "bash", "-c", mr.sweep_script(), app=None, image=None, volumes={}, secrets=[], **options
    )  # raises TypeError if any keyword is not a Sandbox.create parameter


def test_the_image_chain_builds_offline_with_the_real_sdk(local) -> None:
    modal = pytest.importorskip("modal")
    stage = mr.stage_code(Path("a.tar.gz"), Path("stage"), "HEAD")
    image = mr.build_image(modal, parse(), stage)
    assert image is not None  # lazy: constructed, not built; no network call was needed


def test_the_real_sdk_not_found_error_counts_as_a_missing_path() -> None:
    """Pins the class the live listdir raised ("No such file or directory") to the helper."""
    modal = pytest.importorskip("modal")
    error = modal.exception.NotFoundError("No such file or directory")
    assert not isinstance(error, FileNotFoundError)
    assert mr._is_missing_path(error)
