"""Run the night_run sweep on a Modal GPU Sandbox, and follow it from anywhere.

The sweep runs as the entrypoint of a named, detached Modal Sandbox, so closing the laptop does not
stop it. All state lives on a persistent Modal Volume mounted at ``/vol``: ``data/`` (the inputs and
``night_results.csv``), the best-config JSONs, the submissions, the HuggingFace cache and the log.
The sweep is resumable, so a Sandbox that ends (the 24 h cap, a crash, ``stop``) is continued by the
next ``launch``. Billing stops when the Sandbox ends.

    launch   stage the committed code, upload the data once, start the sweep Sandbox (detached)
    status   is the Sandbox running, how did the last one end, configs done, the log tail
    fetch    download results from the Volume into the repo (safe mid-run: the CSV is merged)
    stop     terminate the Sandbox (the Volume is kept; ``launch`` later resumes)

Everything that is provider-neutral (code archive, token forwarding, results merging) is shared with
``lightning_run``. The environment image is built from ``pyproject.toml``'s dependencies and the
``nn`` / ``tfm`` / ``track`` extras, and cached by Modal until those change; the code itself is
mounted at container start, so a code change does not rebuild it.

Auth is Modal's own: ``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET`` (loaded from ``local.machine.env``)
or a prior ``modal token set``. ``--dry-run`` prints the plan and touches neither Modal nor the SDK;
forwarded secrets show as ``***``.
"""

import argparse
import io
import logging
import os
import shutil
import sys
import tarfile
import time
import tomllib
from pathlib import Path

from lightning_run import (
    RESULT_PATHS,
    SECRET_ENV,
    build_code_archive,
    data_tar_sources,
    git_revision,
    install_results,
    marker_sources,
    marker_text,
    token_env,
    unsynced_sources,
    wandb_api_key,
    wandb_env,
)

from cypshift.local_env import load_local_env

logger = logging.getLogger("modal_run")

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

APP_NAME = "cyp-shift"
SANDBOX_NAME = "sweep"
VOLUME_NAME = "cyp-shift-sweep"
#: Where the Volume is mounted, and where the committed code is mounted, inside the Sandbox.
MOUNT = "/vol"
WORKDIR = "/root/cyp-shift"
#: Written (inside the Volume) after a successful data upload; its JSON content lists the data
#: paths the Volume holds, so a later launch ships only what it does not have.
DATA_MARKER = "/data/.cyp_synced"
LOG_FILE = "sweep.log"
#: The sweep's exit status, written by the entry script; absent while it runs or if it was killed.
EXIT_FILE = "sweep.exit"

#: A Sandbox's `timeout` cap, in hours.
MAX_HOURS = 24.0
DEFAULT_GPU = "L4"
#: Worth naming in ``--help``; not an allow-list (Modal validates the string at Sandbox creation).
GPU_HINTS = ("T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100")
#: Extras installed into the image: the torch/chemprop graph models, TabPFN, and W&B.
IMAGE_EXTRAS = ("nn", "tfm", "track")
#: OS libraries the wheels expect that debian-slim lacks (LightGBM's OpenMP; RDKit's X11 stubs).
APT_PACKAGES = ("libgomp1", "libxrender1", "libxext6")


def image_requirements(pyproject: Path) -> list[str]:
    """The pip requirements for the image: the project's dependencies plus :data:`IMAGE_EXTRAS`.

    Read from ``pyproject.toml`` so the image tracks the project; the package itself is not
    installed (``PYTHONPATH`` points at the mounted source), so code changes never rebuild it.
    """
    project = tomllib.loads(pyproject.read_text())["project"]
    requirements = list(project["dependencies"])
    for extra in IMAGE_EXTRAS:
        requirements += project["optional-dependencies"][extra]
    return list(dict.fromkeys(requirements))


def sweep_script(mount: str = MOUNT, workdir: str = WORKDIR, python: str = "python") -> str:
    """The Sandbox entrypoint: link the Volume into the tree, run the sweep, record its exit.

    The sweep writes to repo-relative paths (``data/``, ``submissions/``, ``configs/night_best*``);
    symlinking those into the Volume makes every output durable with no change to the sweep. A real
    directory where a link belongs is removed only if empty, so nothing of value is ever deleted.
    The exit status goes to ``sweep.exit`` so ``status`` can tell a finished sweep from one that was
    killed (no file) or failed (non-zero).
    """
    return f"""set -u
link_dir() {{
  if [ -d "$1" ] && [ ! -L "$1" ]; then
    rmdir "$1" || {{ echo "refusing: $1 is not empty"; exit 1; }}
  fi
  ln -sfn "$2" "$1"
}}
mkdir -p {mount}/data {mount}/hf_cache {mount}/configs {mount}/submissions
cd {workdir}
rm -f {mount}/{EXIT_FILE}
link_dir data {mount}/data
link_dir submissions {mount}/submissions
mkdir -p configs
ln -sf {mount}/configs/night_best.json configs/night_best.json
ln -sf {mount}/configs/night_best_tdi.json configs/night_best_tdi.json
echo "=== $(date -u +%FT%TZ) sweep start" >> {mount}/{LOG_FILE}
{python} -c "import cypshift, torch, chemprop, lightgbm, rdkit; \\
print('imports ok; cuda available:', torch.cuda.is_available())" >> {mount}/{LOG_FILE} 2>&1
{python} scripts/night_run.py >> {mount}/{LOG_FILE} 2>&1
echo $? > {mount}/{EXIT_FILE}
"""


def sandbox_environment(args: argparse.Namespace, revision: str) -> tuple[dict[str, str], dict]:
    """Split what the sweep needs into plain env vars and secrets (never shown unmasked)."""
    plain = {
        "CYP_WORKERS": str(args.workers),
        "CYP_MEMBERS": str(args.members),
        "CYP_JOBS": str(args.jobs),
        "CYP_PLAN": args.plan,
        "CYP_REV": revision,
        "PYTHONPATH": f"{WORKDIR}/src:{WORKDIR}/scripts",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": f"{MOUNT}/hf_cache",
    }
    secret: dict[str, str] = {}
    for name, value in {**wandb_env(args.wandb), **token_env()}.items():
        (secret if name in SECRET_ENV else plain)[name] = value
    return plain, secret


def sandbox_options(args: argparse.Namespace, plain_env: dict[str, str]) -> dict:
    """Keyword arguments for ``Sandbox.create`` that do not need the SDK to build.

    ``runtime`` is passed only when asked for; the SDK's default applies otherwise.
    """
    options = {
        "name": SANDBOX_NAME,
        "gpu": args.gpu,
        "cpu": args.cpu,
        "memory": int(args.memory_gb * 1024),
        "timeout": int(min(args.max_hours, MAX_HOURS) * 3600),
        "workdir": WORKDIR,
        "env": plain_env,
    }
    if args.runtime:
        options["runtime"] = args.runtime
    return options


def parse_status(
    running: bool, exit_text: str | None, results_csv: str | None, log_tail: str
) -> dict:
    """Combine the Sandbox state and the Volume's files into one status record.

    ``outcome`` is ``running`` while the Sandbox is up. Afterwards ``completed`` for exit 0, else
    ``failed`` (non-zero) or ``stopped`` (no exit file: terminated, or hit the 24 h cap); in the
    last two ``launch`` resumes the sweep from the recorded rows.
    """
    if running:
        outcome = "running"
    elif exit_text is None:
        outcome = "stopped"
    else:
        outcome = "completed" if exit_text.strip() == "0" else "failed"
    rows = max(len(results_csv.splitlines()) - 1, 0) if results_csv else 0
    return {"outcome": outcome, "rows": rows, "tail": log_tail}


def plan_launch(args: argparse.Namespace) -> list[str]:
    """The ordered steps ``launch`` takes (the body of ``launch --dry-run``). Pure: no SDK/cloud.

    Secret values are shown as ``***``.
    """
    revision = args.rev or git_revision(args.ref)
    plain, secret = sandbox_environment(args, revision)
    options = sandbox_options(args, plain)
    return [
        f"git archive {args.ref} -> {args.archive} -> {args.stage}  (revision {revision})",
        f"image: debian-slim python {args.python} + apt {' '.join(APT_PACKAGES)} + pip "
        f"({len(image_requirements(PYPROJECT))} requirements incl. extras "
        f"{','.join(IMAGE_EXTRAS)}); code mounted at {WORKDIR}; cached by Modal",
        f"abort if Sandbox {SANDBOX_NAME!r} is already running in app {args.app!r}",
        f"volume {args.volume!r}: upload {' '.join(data_tar_sources(args.data_external))} unless "
        f"the marker {DATA_MARKER} already records it (--refresh-data resends)",
        f"Sandbox.create(bash -c <sweep script>, volumes={{{MOUNT!r}: {args.volume}}}, "
        + ", ".join(f"{key}={value!r}" for key, value in options.items())
        + (", secrets=[" + ", ".join(f"{name}=***" for name in secret) + "]" if secret else "")
        + ")",
        "wait for it, then fetch"
        if args.wait
        else "return immediately; follow with `status`, download with `fetch`",
    ]


def _require_modal() -> object:
    """Import the Modal SDK, or exit with install/auth guidance."""
    try:
        import modal
    except ImportError as error:
        raise SystemExit(
            "modal_run needs the Modal SDK: pip install 'modal>=1.0' (or 'cypshift[modal]'), then "
            "set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in local.machine.env (tokens: "
            "modal.com/settings/tokens) or run `modal token set`."
        ) from error
    return modal


def stage_code(archive: Path, stage: Path, ref: str) -> Path:
    """Export the committed tree at ``ref`` into ``stage`` (replaced each launch)."""
    build_code_archive(archive, ref)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(stage, filter="data")
    return stage


def build_image(modal: object, args: argparse.Namespace, stage: Path) -> object:
    """The sweep image: dependencies baked (cached), committed code mounted at start."""
    return (
        modal.Image.debian_slim(python_version=args.python)
        .apt_install(*APT_PACKAGES)
        .pip_install(*image_requirements(stage / "pyproject.toml"))
        .add_local_dir(stage, WORKDIR)
    )


def _is_missing_path(error: Exception) -> bool:
    """Whether ``error`` is the SDK's "no such path in the Volume". Verified against modal 1.6.2:
    ``listdir`` / ``iterdir`` raise ``modal.exception.NotFoundError("No such file or directory")``
    (a ``GRPCError``, not a ``FileNotFoundError``) while ``read_file`` raises the builtin. Matched
    by class name so the helper needs no SDK import; a missing *Volume* raises the same class and
    is ruled out beforehand by :func:`open_volume`."""
    return isinstance(error, FileNotFoundError) or type(error).__name__ == "NotFoundError"


def open_volume(modal: object, name: str) -> object:
    """An existing Volume, resolved now: a missing one is a SystemExit here, so a later
    ``NotFoundError`` on a path can only mean the path is absent."""
    volume = modal.Volume.from_name(name)
    try:
        volume.hydrate()
    except Exception as error:
        if not _is_missing_path(error):
            raise
        raise SystemExit(f"Volume {name!r} does not exist yet: run `launch` first") from error
    return volume


def sync_data(volume: object, args: argparse.Namespace) -> bool:
    """Upload the data paths the Volume does not hold yet; True if it uploaded anything.

    The marker records which paths were shipped, so a later launch that adds ``--data-external``
    to an earlier raw-only upload ships just the external caches; ``--refresh-data`` resends all.
    A listing failure other than a clean not-found propagates (see :func:`read_text`): a flaky
    connection must not read as "nothing synced" and re-send 500 MB.
    """
    requested = data_tar_sources(args.data_external)
    recorded = read_text(volume, DATA_MARKER)
    todo = unsynced_sources(requested, recorded, args.refresh_data)
    if not todo:
        logger.info("data already on the volume (%s); not re-uploading", ", ".join(requested))
        return False
    logger.info("uploading %s to the volume", " + ".join(todo))
    with volume.batch_upload(force=True) as batch:
        for source in todo:
            batch.put_directory(source, f"/{source}")
        batch.put_file(
            io.BytesIO(marker_text(marker_sources(recorded) | set(requested)).encode()),
            DATA_MARKER,
        )
    return True


def running_sandbox(modal: object, app_name: str) -> object | None:
    """The running sweep Sandbox, or None. Names are unique among running Sandboxes in an app."""
    try:
        return modal.Sandbox.from_name(app_name, SANDBOX_NAME)
    except modal.exception.NotFoundError:
        return None


def read_text(volume: object, path: str) -> str | None:
    """A small text file from the Volume, or None if it is not there (yet)."""
    try:
        return b"".join(volume.read_file(path)).decode(errors="replace")
    except Exception as error:
        if not _is_missing_path(error):
            raise
        return None


def launch(args: argparse.Namespace, modal: object | None = None) -> object:
    """Start (or resume) the sweep Sandbox, detached. Returns the Sandbox."""
    if args.wandb and not wandb_api_key():
        # Before any Modal call: a detached remote run with no key could not authenticate.
        raise SystemExit(
            "--wandb needs a W&B key to forward: run `wandb login`, or set WANDB_API_KEY "
            "(e.g. in local.machine.env)"
        )
    modal = modal or _require_modal()
    revision = args.rev or git_revision(args.ref)
    app = modal.App.lookup(args.app, create_if_missing=True)
    if running_sandbox(modal, args.app) is not None:
        raise SystemExit(
            f"a sweep Sandbox is already running in app {args.app!r}; use `status` to follow it, "
            "`fetch` to download, or `stop` to end it"
        )
    stage = stage_code(Path(args.archive), Path(args.stage), args.ref)
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)
    sync_data(volume, args)

    plain, secret = sandbox_environment(args, revision)
    options = sandbox_options(args, plain)
    if secret:
        options["secrets"] = [modal.Secret.from_dict(secret)]
    sandbox = modal.Sandbox.create(
        "bash",
        "-c",
        sweep_script(),
        app=app,
        image=build_image(modal, args, stage),
        volumes={MOUNT: volume},
        **options,
    )
    print(f"sweep started on {args.gpu} (revision {revision}); Sandbox {sandbox.object_id}")
    print(f"  dashboard: {sandbox.get_dashboard_url()}")
    print(f"  follow:    python scripts/modal_run.py status --app {args.app}")
    print(f"  download:  python scripts/modal_run.py fetch  --volume {args.volume}")
    if args.wait:
        wait_and_fetch(args, modal, volume)
    return sandbox


def live_log(sandbox: object, lines: int = 5) -> str | None:
    """The log tail read *inside* the running Sandbox, or None if that is not possible.

    The Volume copy of the log lags: the sweep holds ``sweep.log`` open for its whole run and a
    client sees it only once it is closed (observed live: absent from the Volume listing for 10+
    minutes while the Sandbox already held 14 KB), whereas ``night_results.csv`` is closed after
    every write and shows up promptly."""
    try:
        process = sandbox.exec("tail", "-n", str(lines), f"{MOUNT}/{LOG_FILE}", timeout=30)
        text = process.stdout.read()
        process.wait()
    except Exception as error:
        logger.warning(
            "could not read the log in the Sandbox (%s); using the Volume copy",
            type(error).__name__,
        )
        return None
    return text or None


def read_status(modal: object, args: argparse.Namespace) -> dict:
    """The Sandbox and Volume state as a :func:`parse_status` record."""
    volume = open_volume(modal, args.volume)
    sandbox = running_sandbox(modal, args.app)
    log = (
        (live_log(sandbox) if sandbox is not None else None)
        or read_text(volume, f"/{LOG_FILE}")
        or ""
    )
    return parse_status(
        sandbox is not None,
        read_text(volume, f"/{EXIT_FILE}"),
        read_text(volume, "/data/night_results.csv"),
        "\n".join(log.splitlines()[-5:]),
    )


def status(args: argparse.Namespace, modal: object | None = None) -> dict:
    modal = modal or _require_modal()
    record = read_status(modal, args)
    print(f"sweep: {record['outcome']}  |  configs recorded: {record['rows']}")
    if record["outcome"] in ("failed", "stopped"):
        print("  `launch` resumes it from the recorded rows")
    if record["tail"]:
        print("--- log tail ---")
        print(record["tail"])
    return record


def fetch(args: argparse.Namespace, modal: object | None = None) -> int:
    """Download the result files from the Volume into the repo; returns result rows added."""
    modal = modal or _require_modal()
    volume = open_volume(modal, args.volume)
    staging = Path(args.archive_dir) / "modal_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    found = 0
    for remote in RESULT_PATHS:
        try:
            entries = [e for e in volume.iterdir(f"/{remote}", recursive=True) if _is_file(e)]
        except Exception as error:
            if not _is_missing_path(error):
                raise
            continue  # this output has not been written (a members-screening run skips some)
        for entry in entries:
            target = staging / entry.path.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"".join(volume.read_file(entry.path)))
            found += 1
    added = install_results(staging, Path.cwd())
    shutil.rmtree(staging)
    print(f"fetched {found} files; {added} new result rows merged into data/night_results.csv")
    return added


def _is_file(entry: object) -> bool:
    return getattr(entry.type, "name", str(entry.type)) == "FILE"


def wait_and_fetch(args: argparse.Namespace, modal: object, volume: object) -> None:
    """Poll until the Sandbox ends, then fetch (also fetching on Ctrl-C, since results persist)."""
    try:
        while running_sandbox(modal, args.app) is not None:
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("interrupted; the sweep keeps running remotely")
        return
    fetch(args, modal)


def stop(args: argparse.Namespace, modal: object | None = None) -> None:
    modal = modal or _require_modal()
    sandbox = running_sandbox(modal, args.app)
    if sandbox is None:
        print("no sweep Sandbox is running")
        return
    sandbox.terminate(wait=True)
    print("sweep Sandbox terminated (the Volume is kept; `launch` resumes from the recorded rows)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--app", default=os.environ.get("CYP_MODAL_APP", APP_NAME))
    common.add_argument("--volume", default=os.environ.get("CYP_MODAL_VOLUME", VOLUME_NAME))
    common.add_argument("--archive-dir", default="scratch", help="where downloads are staged")

    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("launch", parents=[common], help="start or resume the sweep")
    run.add_argument("--gpu", default=DEFAULT_GPU, help=f"Modal GPU, e.g. {', '.join(GPU_HINTS)}")
    run.add_argument(
        "--cpu", type=float, default=8.0, help="CPU cores (also the dataloader workers)"
    )
    run.add_argument("--memory-gb", type=float, default=32.0, help="container memory, GiB")
    run.add_argument(
        "--max-hours",
        type=float,
        default=MAX_HOURS,
        help=f"Sandbox timeout, capped at {MAX_HOURS:g}",
    )
    run.add_argument("--runtime", choices=("gvisor", "vm"), default=None, help="Sandbox runtime")
    run.add_argument("--python", default="3.12", help="image Python version")
    run.add_argument("--workers", type=int, default=8, help="remote CYP_WORKERS")
    run.add_argument(
        "--members",
        type=int,
        default=1,
        help="remote CYP_MEMBERS (1 screens architectures fast; 0 uses each family's default)",
    )
    run.add_argument(
        "--plan",
        choices=("full", "focus"),
        default="full",
        help="remote CYP_PLAN: the whole queue, or the staged focus plan",
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="remote CYP_JOBS: configurations trained at the same time on the one GPU",
    )
    run.add_argument("--ref", default="HEAD", help="git ref to archive and run")
    run.add_argument(
        "--rev", default=None, help="override the CYP_REV namespace (default: ref SHA)"
    )
    run.add_argument("--archive", default="scratch/modal_code.tar.gz")
    run.add_argument(
        "--stage", default="scratch/modal_code", help="staged copy mounted in the Sandbox"
    )
    run.add_argument(
        "--data-external", action="store_true", help="also upload data/external caches"
    )
    run.add_argument("--refresh-data", action="store_true", help="re-upload the data")
    run.add_argument("--wandb", action="store_true", help="follow the sweep online in W&B")
    run.add_argument("--wait", action="store_true", help="block until the sweep ends, then fetch")
    run.add_argument("--poll", type=int, default=300, help="seconds between --wait polls")
    run.add_argument("--dry-run", action="store_true", help="print the plan; touch nothing")
    commands.add_parser("status", parents=[common], help="Sandbox state, progress, log tail")
    commands.add_parser("fetch", parents=[common], help="download and merge results")
    commands.add_parser("stop", parents=[common], help="terminate the sweep Sandbox")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_local_env()
    args = build_parser().parse_args(argv)
    if args.command == "launch":
        if args.dry_run:
            print("Modal sweep launch plan:")
            for index, step in enumerate(plan_launch(args), start=1):
                print(f"  {index}. {step}")
            return
        launch(args)
    elif args.command == "status":
        status(args)
    elif args.command == "fetch":
        fetch(args)
    else:
        stop(args)


if __name__ == "__main__":
    sys.exit(main())
