"""Run the night_run sweep on a persistent Lightning AI Studio, and follow it from anywhere.

A Lightning Studio keeps its filesystem across stop/start, so this treats it as a long-lived
workstation rather than a throwaway job: the environment, the data and ``data/night_results.csv``
all survive, and the sweep (which is resumable) simply continues where it left off. The sweep runs
*detached* on the Studio, so closing the laptop does not kill it; ``status`` and ``fetch`` reattach.

    launch   start/reuse the Studio, sync the code, ensure the env, start the sweep detached
    status   is the Studio up, is the sweep alive, how many configs are done, the log tail
    fetch    download results into the repo (safe mid-run: a partial CSV is merged, never clobbered)
    stop     stop the Studio (its filesystem is kept; ``launch`` later resumes)

``launch`` is idempotent. It reuses a running Studio (the SDK's ``start`` raises on one), always
refreshes the small code tarball, uploads the data only the first time (or with ``--refresh-data``),
and runs the check-first ``./setup.sh`` every time -- a few seconds once the env exists. The env is
built inside the project directory (``ENV_PREFIX``) so the interpreter path is fixed and persists.

The sweep's run identity (``REV``) normally comes from ``git``; a ``git archive`` has no ``.git``,
so this passes the archived commit's SHA as ``CYP_REV``. Without that, every code version would
share the ``nogit`` namespace and a resume would silently reuse results from different code.

Auth is the Lightning SDK's own (``LIGHTNING_USER_ID`` + ``LIGHTNING_API_KEY``, or a prior
``lightning login``). ``main`` first loads the gitignored ``local.machine.env`` into the environment
(``cypshift.local_env``; an exported variable wins), which is also where the tokens forwarded to the
sweep come from. Nothing here stores a credential, and ``--dry-run`` shows the forwarded ones as
``***``. It prints the plan and touches neither the cloud nor the SDK. Remote paths are relative to
the Studio root, which is where both the shell and the file API start.
"""

import argparse
import json
import logging
import netrc
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from cypshift.local_env import load_local_env

logger = logging.getLogger("lightning_run")

#: The sweep's outputs, relative to the repo root. Only those that exist on the remote are
#: archived for download (a members-screening run writes the same paths as a full run).
RESULT_PATHS = (
    "data/night_results.csv",
    "data/run_bundles",
    "data/focus_path.json",
    "data/focus_best.json",
    "data/focus_blends.csv",
    "configs/night_best.json",
    "configs/night_best_tdi.json",
    "submissions/night_submission_regression.csv",
    "submissions/night_submission_classification.csv",
)

#: Machines worth naming in ``--help``. Not an allow-list: the choice is validated against the SDK's
#: own ``Machine`` members at launch, so a newly offered machine needs no code change here.
MACHINE_HINTS = ("CPU", "T4", "L4", "L40S", "A100", "H100")

#: Remote working directory, relative to the Studio root (the file API resolves paths there; a
#: shell ``~`` is not expanded by it).
DEFAULT_REMOTE_DIR = "cyp-shift"
#: The env is built inside the project directory, so it persists with it and the interpreter path
#: is known without asking conda.
REMOTE_ENV = ".env"
DEFAULT_REMOTE_PYTHON = f"./{REMOTE_ENV}/bin/python"
#: Written after the first successful data sync, so later launches skip the upload.
DATA_MARKER = "data/.cyp_synced"
PID_FILE = "sweep.pid"
LOG_FILE = "sweep.log"


def git_revision(ref: str = "HEAD") -> str:
    """Short SHA of ``ref`` -- the revision a ``git archive`` of it actually contains."""
    return subprocess.check_output(["git", "rev-parse", "--short", ref], text=True).strip()


def build_code_archive(archive_path: Path, ref: str = "HEAD") -> Path:
    """Archive the tracked tree at ``ref`` into ``archive_path`` (a .tar.gz).

    ``git archive`` ships only tracked files, so data, the ``.git`` directory and the gitignored
    working configs never travel -- the remote gets exactly the committed code the sweep runs.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(archive_path), ref],
        check=True,
    )
    return archive_path


def data_tar_sources(include_external: bool) -> list[str]:
    """The data paths to ship: the core ``data/raw`` always, the auxiliary caches on request.

    ``data/raw`` holds the challenge files; ``data/external`` holds the auxiliary caches
    (octant/tox21/veith/chembl/sair/novartis, embeddings, the CheMeleon encoder). Some of those
    download on first use remotely, but the local-only caches (octant, novartis) must be shipped.
    """
    sources = ["data/raw"]
    if include_external:
        sources.append("data/external")
    return sources


def marker_sources(text: str | None) -> set[str]:
    """The data sources a sync marker records as already on the remote.

    The marker is JSON ``{"sources": [...]}``. A marker with any other content (the bare ``ok`` /
    empty file earlier launchers wrote, which did not say what was shipped) counts as the core
    ``data/raw`` only -- every sync has always shipped that -- so a later request for the
    auxiliary caches re-sends them rather than silently training on a smaller panel. ``None``
    (no marker) is nothing synced.
    """
    if text is None:
        return set()
    try:
        sources = json.loads(text)["sources"]
        return {str(source) for source in sources}
    except (ValueError, KeyError, TypeError):
        return {"data/raw"}


def marker_text(sources: set[str] | list[str]) -> str:
    """The marker file content for ``sources`` (sorted, so it is stable)."""
    return json.dumps({"sources": sorted(set(sources))})


def unsynced_sources(requested: list[str], marker: str | None, refresh: bool) -> list[str]:
    """The requested data paths still to ship: all of them with ``refresh``, else those the marker
    does not record. A raw-only sync followed by a request that adds ``data/external`` therefore
    ships the external caches instead of skipping on the mere presence of the marker."""
    if refresh:
        return list(requested)
    synced = marker_sources(marker)
    return [source for source in requested if source not in synced]


def wandb_api_key() -> str | None:
    """The local W&B key: ``WANDB_API_KEY`` if set, else the one ``wandb login`` stored in
    ``~/.netrc``. ``wandb login`` does not export the variable, so without this fallback a logged-in
    machine would forward no key and the detached remote run could not authenticate."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    try:
        entry = netrc.netrc().authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        return None
    return entry[2] if entry and entry[2] else None


def wandb_env(enabled: bool) -> dict[str, str]:
    """The W&B environment to forward to the remote sweep when ``--wandb`` is on.

    Turns the sweep's ``CYP_WANDB`` gate on and carries the API key (for non-interactive auth) and
    whichever of ``WANDB_PROJECT`` / ``WANDB_ENTITY`` are set locally (where the run lands), so the
    remote run reports to the same account and project without an interactive ``wandb login``.
    """
    if not enabled:
        return {}
    env = {"CYP_WANDB": "1"}
    key = wandb_api_key()
    if key:
        env["WANDB_API_KEY"] = key
    for name in ("WANDB_PROJECT", "WANDB_ENTITY"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


#: Forwarded variables whose values are never printed (``--dry-run``, logs).
SECRET_ENV = ("WANDB_API_KEY", "TABPFN_TOKEN", "HF_TOKEN")


def hf_token() -> str | None:
    """The local HuggingFace token: ``HF_TOKEN`` if set, else the one ``hf auth login`` stored.

    ``huggingface_hub.get_token`` resolves both (and honours ``HF_HOME``). As with the W&B key,
    logging in exports nothing, so without this a logged-in machine would forward no token.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token() or None


def token_env() -> dict[str, str]:
    """The sweep tokens available locally: ``TABPFN_TOKEN`` (the ``tfm_*`` configs) and the
    HuggingFace token (gated assets). Unlike W&B these need no flag: without them the configs that
    use them fail and are skipped, which a remote sweep would only discover hours in."""
    env = {}
    if os.environ.get("TABPFN_TOKEN"):
        env["TABPFN_TOKEN"] = os.environ["TABPFN_TOKEN"]
    token = hf_token()
    if token:
        env["HF_TOKEN"] = token
    return env


def setup_command(remote_dir: str) -> str:
    """Build or verify the env inside the project directory. Idempotent: ``setup.sh`` is
    check-first, so once the env exists this is a few seconds of import probes."""
    return f"cd {remote_dir} && ENV_PREFIX=$PWD/{REMOTE_ENV} bash ./setup.sh"


def detached_sweep_command(
    remote_dir: str,
    python: str,
    workers: int,
    members: int,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Start the sweep in the background and record its pid, so it outlives this connection.

    ``nohup`` with redirected output keeps it alive after the session ends, and the pid is written
    from inside the same subshell that backgrounds it (so ``$!`` is the sweep's pid and the ``cd``
    applies). ``extra_env`` carries ``CYP_REV`` and any W&B variables. ``nohup``/``env`` exec in
    place, so the recorded pid stays the python process's.
    """
    env = {"CYP_WORKERS": str(workers), "CYP_MEMBERS": str(members), **(extra_env or {})}
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return (
        f"cd {remote_dir} && (nohup env {assignments} {python} scripts/night_run.py "
        f"> {LOG_FILE} 2>&1 < /dev/null & echo $! > {PID_FILE})"
    )


def status_command(remote_dir: str) -> str:
    """One shell snippet reporting the sweep's liveness, progress and log tail as ``KEY=value``.

    Liveness checks the process *is* the sweep (not just that the pid exists), so a pid left over
    from before a Studio restart and reused by something else is not mistaken for a live sweep. It
    reads ``/proc/<pid>/cmdline``: ``ps -o args=`` truncates to 80 columns when its output is not a
    terminal, which cut ``night_run.py`` off a long interpreter path and made a live sweep look
    stopped.
    """
    return (
        f"cd {remote_dir} 2>/dev/null || {{ echo PROJECT=missing; exit 0; }}; "
        "echo PROJECT=present; "
        f'if [ -f {PID_FILE} ] && kill -0 "$(cat {PID_FILE})" 2>/dev/null '
        f"&& tr '\\0' ' ' < /proc/\"$(cat {PID_FILE})\"/cmdline | grep -q night_run.py; "
        f"then echo SWEEP=running; else echo SWEEP=stopped; fi; "
        f"if [ -f data/night_results.csv ]; then "
        f"echo ROWS=$(( $(wc -l < data/night_results.csv) - 1 )); else echo ROWS=0; fi; "
        f"echo ---; tail -n 5 {LOG_FILE} 2>/dev/null; true"
    )


def parse_status(output: str) -> dict:
    """Parse :func:`status_command` output into ``project`` / ``running`` / ``rows`` / ``tail``."""
    head, _, tail = output.partition("---")
    fields = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    return {
        "project": fields.get("PROJECT") == "present",
        "running": fields.get("SWEEP") == "running",
        "rows": int(fields.get("ROWS", 0) or 0),
        "tail": tail.strip(),
    }


def result_tar_command(remote_dir: str, archive_name: str) -> str:
    """A remote command that tars only the result paths that exist, into ``archive_name``.

    The run may not write every path (an empty TDI track leaves no TDI submission), so each is
    added only ``if`` present -- ``tar`` on a missing path would fail the whole download.
    """
    existing = " ".join(f"$(test -e {path} && echo {path})" for path in RESULT_PATHS)
    return f"cd {remote_dir} && tar czf {archive_name} {existing}"


def merge_results(local_csv: Path, incoming_csv: Path) -> int:
    """Append the incoming sweep rows to ``local_csv`` without losing local ones.

    A plain extract would overwrite the local results with the Studio's, discarding any rows from
    runs made here. Rows are keyed by their full content, so re-fetching a partial CSV mid-run and
    again at the end adds only what is new. Returns the number of rows the merge added.
    """
    incoming = pd.read_csv(incoming_csv)
    if local_csv.exists():
        existing = pd.read_csv(local_csv)
        merged = pd.concat([existing, incoming], ignore_index=True).drop_duplicates()
        added = len(merged) - len(existing)
    else:
        merged, added = incoming, len(incoming)
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(local_csv, index=False)
    return added


def install_results(staging: Path, repo_root: Path) -> int:
    """Copy a staged results tree into ``repo_root``, merging the results CSV.

    Bundles, best-config JSONs and submissions are copied in place (bundles are namespaced by
    revision, so they cannot collide); ``data/night_results.csv`` goes through
    :func:`merge_results`. Provider-neutral: the Studio path stages a tarball, the Modal path stages
    files read from its volume. Returns the number of result rows added.
    """
    added = 0
    for source in staging.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(staging)
        target = repo_root / relative
        if relative == Path("data/night_results.csv"):
            added = merge_results(target, source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return added


def extract_results(archive: Path, repo_root: Path) -> int:
    """Unpack a downloaded results archive into ``repo_root``, merging the results CSV.

    Returns the number of result rows added (see :func:`install_results`).
    """
    staging = repo_root / "scratch" / "lightning_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(staging, filter="data")  # "data" filter: no absolute paths or escapes
    added = install_results(staging, repo_root)
    shutil.rmtree(staging)
    return added


def studio_is_running(studio: object) -> bool:
    """Whether the Studio is up. Compared by status name, so no SDK import is needed."""
    status = studio.status
    return getattr(status, "name", str(status)).lower() == "running"


def _require_sdk() -> tuple[type, type]:
    """Import the Lightning SDK, or exit with install/auth guidance. Returns (Studio, Machine)."""
    try:
        from lightning_sdk import Machine, Studio
    except ImportError as error:
        raise SystemExit(
            "lightning_run needs the Lightning SDK: pip install 'cypshift[cloud]', then set "
            "LIGHTNING_USER_ID and LIGHTNING_API_KEY (or run `lightning login`)."
        ) from error
    return Studio, Machine


def resolve_machine(name: str) -> object:
    """Map a ``--machine`` name to the SDK's ``Machine`` member, validated against the SDK."""
    _, machine_enum = _require_sdk()
    available = sorted(n for n in dir(machine_enum) if n.isupper())
    key = name.upper()
    if key not in available:
        raise SystemExit(f"unknown --machine {name!r}; the SDK offers: {', '.join(available)}")
    return getattr(machine_enum, key)


def _describe_failure(error: Exception) -> str:
    """One short line for an SDK failure. Cloud errors can carry a whole HTML page as their text,
    so only the type, the HTTP status when there is one, and a clipped first line are kept."""
    status = getattr(error, "status", None)
    first = (str(error).strip().splitlines() or [""])[0][:120]
    return f"{type(error).__name__}{f' {status}' if status else ''}: {first}"


def teamspace_hint() -> str:
    """Where this account's teamspaces live, for the unresolved-teamspace error.

    Lists the logged-in user's teamspaces and each organization's, labelled by owner kind so it is
    clear whether to pass ``--user`` or ``--org``. Best effort, but never silent: a failure is
    reported in the hint (a regional block or bad credentials look the same as "nothing to list"
    otherwise) and never replaces the error it explains. ``_get_authed_user`` is the SDK's own
    (private) resolver of the logged-in user.
    """
    try:
        from lightning_sdk.utils.resolve import _get_authed_user

        user = _get_authed_user()
        owners = [("user", user), *[("org", org) for org in user.organizations]]
    except Exception as error:  # noqa: BLE001 -- a hint must never replace the error it explains
        return f" (could not look up your account to suggest one: {_describe_failure(error)})"
    parts = []
    for kind, owner in owners:
        try:
            names = [ts.name for ts in owner.teamspaces]
        except Exception as error:  # noqa: BLE001 -- keep the owners that did list
            parts.append(f"{kind} {owner.name!r}: unreadable ({_describe_failure(error)})")
            continue
        parts.append(f"{kind} {owner.name!r}: {', '.join(repr(n) for n in names) or '(none)'}")
    return " Your account's teamspaces -- " + "; ".join(parts) + "."


def open_studio(args: argparse.Namespace) -> object:
    """The named Studio, created on first use. Opening it does not start it.

    The SDK needs a teamspace name and its owner (a user or an org). Both can come from the flags
    or from the SDK's own ``LIGHTNING_TEAMSPACE`` / ``LIGHTNING_USERNAME`` / ``LIGHTNING_ORG``
    variables; when none resolves it raises a bare ``ValueError``, which is turned into guidance
    here, before any Studio exists.
    """
    studio_cls, _ = _require_sdk()
    try:
        return studio_cls(
            name=args.studio,
            teamspace=args.teamspace,
            user=args.user,
            org=args.org,
            create_ok=True,
        )
    except ValueError as error:
        if "teamspace" not in str(error).lower():
            raise
        raise SystemExit(
            "no Lightning teamspace to put the Studio in: pass --teamspace NAME (plus --org ORG if "
            "an organization owns it, else --user USERNAME), or set LIGHTNING_TEAMSPACE and "
            "LIGHTNING_USERNAME (or LIGHTNING_ORG) in local.machine.env." + teamspace_hint()
        ) from error


def ensure_running(
    studio: object, args: argparse.Namespace, resolver: Callable = resolve_machine
) -> None:
    """Start the Studio unless it is already up (the SDK's ``start`` raises on a running one)."""
    if studio_is_running(studio):
        logger.info("Studio %s is already running; reusing it (--machine ignored)", args.studio)
        return
    max_runtime = int(args.max_hours * 3600) if args.max_hours else None
    logger.info("starting Studio %s on %s", args.studio, args.machine)
    studio.start(resolver(args.machine), max_runtime=max_runtime)


def upload_checked(studio: object, local: Path, remote: str) -> None:
    """Upload one file and confirm the shell sees it where the file API put it.

    The file API and the shell are separate views of the Studio; if their roots ever differ, an
    unchecked upload would succeed and the next command would fail confusingly; failing here says
    so.
    """
    studio.upload_file(str(local), remote)
    try:
        studio.run(f"test -f {shlex.quote(remote)}")
    except RuntimeError as error:
        raise SystemExit(
            f"uploaded {local} to {remote!r} but the shell cannot see it there; the file API root "
            "and the shell directory differ -- set --remote-dir to an absolute path"
        ) from error


def sync_code(studio: object, args: argparse.Namespace, archive: Path) -> None:
    """Refresh the code on the Studio. Small, so done on every launch."""
    remote_dir = args.remote_dir
    studio.run(f"mkdir -p {remote_dir}")
    upload_checked(studio, archive, f"{remote_dir}/code.tar.gz")
    studio.run(f"cd {remote_dir} && tar xzf code.tar.gz")


def sync_data(studio: object, args: argparse.Namespace, data_archive: Path) -> bool:
    """Upload the data paths the Studio does not yet have. Returns whether it uploaded.

    The marker records which paths were shipped, so a later launch that adds ``--data-external``
    to an earlier raw-only sync ships just the external caches; ``--refresh-data`` resends all.
    """
    remote_dir = args.remote_dir
    requested = data_tar_sources(args.data_external)
    text, code = studio.run_with_exit_code(f"cat {remote_dir}/{DATA_MARKER}")
    recorded = text if code == 0 else None
    todo = unsynced_sources(requested, recorded, args.refresh_data)
    if not todo:
        logger.info("data already on the Studio (%s); skipping upload", ", ".join(requested))
        return False
    data_archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "czf", str(data_archive), *todo], check=True)
    upload_checked(studio, data_archive, f"{remote_dir}/data.tar.gz")
    marker = marker_text(marker_sources(recorded) | set(requested))
    studio.run(
        f"cd {remote_dir} && tar xzf data.tar.gz && printf %s {shlex.quote(marker)} > {DATA_MARKER}"
    )
    return True


def sweep_env(args: argparse.Namespace, revision: str) -> dict[str, str]:
    """Environment forwarded to the sweep: the archived revision, the sweep tokens, and W&B when
    enabled."""
    return {"CYP_REV": revision, **wandb_env(args.wandb), **token_env()}


def read_status(studio: object, remote_dir: str) -> dict:
    return parse_status(studio.run(status_command(remote_dir)))


def launch(
    args: argparse.Namespace,
    studio: object | None = None,
    resolver: Callable = resolve_machine,
) -> None:
    """Start (or resume) the sweep on the persistent Studio, detached."""
    if args.wandb and not wandb_api_key():
        # Before any cloud call: a detached remote run with no key could not authenticate, and
        # would only show it in a log nobody is watching.
        raise SystemExit(
            "--wandb needs a W&B key to forward: run `wandb login`, or set WANDB_API_KEY "
            "(e.g. in local.machine.env)"
        )
    studio = studio or open_studio(args)
    revision = args.rev or git_revision(args.ref)
    archive = build_code_archive(Path(args.archive), args.ref)
    ensure_running(studio, args, resolver)

    current = read_status(studio, args.remote_dir)
    if current["running"]:
        raise SystemExit(
            f"a sweep is already running on {args.studio} ({current['rows']} configs recorded); "
            "use `status` to follow it or `fetch` to download its results"
        )
    sync_code(studio, args, archive)
    sync_data(studio, args, Path(args.data_archive))
    if args.setup:
        logger.info("checking the remote env (setup.sh is check-first)")
        logger.info("%s", studio.run(setup_command(args.remote_dir)))
    studio.run(
        detached_sweep_command(
            args.remote_dir,
            args.remote_python,
            args.workers,
            args.members,
            sweep_env(args, revision),
        )
    )
    logger.info(
        "sweep started detached on %s at revision %s. Follow with:\n"
        "  python scripts/lightning_run.py status --studio %s\n"
        "  python scripts/lightning_run.py fetch  --studio %s",
        args.studio,
        revision,
        args.studio,
        args.studio,
    )
    if args.wait:
        wait_and_fetch(studio, args)


def wait_and_fetch(studio: object, args: argparse.Namespace, sleep: Callable = time.sleep) -> None:
    """Block until the sweep ends, then fetch; stop the Studio afterwards only on request."""
    while read_status(studio, args.remote_dir)["running"]:
        sleep(args.poll)
    fetch(args, studio)
    if args.stop_after:
        stop(args, studio)


def fetch(args: argparse.Namespace, studio: object | None = None) -> None:
    """Download the results (partial is fine) and merge them into the repo."""
    studio = studio or open_studio(args)
    if not studio_is_running(studio):
        raise SystemExit(
            f"Studio {args.studio} is not running; `launch` starts it, or start it in the UI"
        )
    remote_archive = f"{args.remote_dir}/results.tar.gz"
    studio.run(result_tar_command(args.remote_dir, "results.tar.gz"))
    local = Path(args.archive_dir) / "results.tar.gz"
    local.parent.mkdir(parents=True, exist_ok=True)
    studio.download_file(remote_archive, str(local))
    added = extract_results(local, Path.cwd())
    logger.info("results merged into the repo (%d new rows in data/night_results.csv)", added)


def status(args: argparse.Namespace, studio: object | None = None) -> None:
    studio = studio or open_studio(args)
    up = studio_is_running(studio)
    print(f"Studio {args.studio}: {getattr(studio.status, 'name', studio.status)}")
    if not up:
        print("  not running -- `launch` starts it and resumes the sweep from the saved results")
        return
    current = read_status(studio, args.remote_dir)
    if not current["project"]:
        print(f"  no project at {args.remote_dir} yet -- run `launch`")
        return
    print(f"  sweep: {'RUNNING' if current['running'] else 'not running'}")
    print(f"  configs recorded: {current['rows']}")
    if current["tail"]:
        print("  log tail:\n    " + current["tail"].replace("\n", "\n    "))


def stop(args: argparse.Namespace, studio: object | None = None) -> None:
    """Stop the Studio. Its filesystem is kept, so ``launch`` resumes where the sweep left off."""
    studio = studio or open_studio(args)
    if not studio_is_running(studio):
        print(f"Studio {args.studio} is already stopped")
        return
    if read_status(studio, args.remote_dir)["running"] and not args.force:
        raise SystemExit("a sweep is still running; fetch first, or pass --force to stop it anyway")
    studio.stop()
    print(f"Studio {args.studio} stopped (filesystem kept)")


def plan_launch(args: argparse.Namespace) -> list[str]:
    """The ordered steps ``launch`` takes -- the body of ``launch --dry-run``.

    Pure: built from the arguments alone, with no cloud call and no SDK import. The API key is
    shown as ``***``, never its value.
    """
    revision = args.rev or git_revision(args.ref)
    redacted = {
        key: ("***" if key in SECRET_ENV else value)
        for key, value in sweep_env(args, revision).items()
    }
    steps = [
        f"git archive {args.ref} -> {args.archive}  (revision {revision})",
        f"open Studio {args.studio!r}; start it on {args.machine} only if not already running",
        f"abort if a sweep is already running in {args.remote_dir}",
        f"upload code tarball -> {args.remote_dir}/code.tar.gz, extract (every launch)",
        f"data: tar {' '.join(data_tar_sources(args.data_external))}, upload and extract what the "
        f"marker {DATA_MARKER} does not record (--refresh-data resends all)",
    ]
    if args.setup:
        steps.append("remote: " + setup_command(args.remote_dir) + "  (check-first, idempotent)")
    steps.append(
        "remote (detached): "
        + detached_sweep_command(
            args.remote_dir, args.remote_python, args.workers, args.members, redacted
        )
    )
    steps.append(
        "wait for it, then fetch" + (" and stop the Studio" if args.stop_after else "")
        if args.wait
        else "return immediately; follow with `status`, download with `fetch`"
    )
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--studio", default=os.environ.get("CYP_LAI_STUDIO", "cyp-shift"))
    common.add_argument("--teamspace", default=os.environ.get("CYP_LAI_TEAMSPACE"))
    common.add_argument(
        "--user", default=None, help="teamspace owner, if a user (SDK default: LIGHTNING_USERNAME)"
    )
    common.add_argument(
        "--org", default=None, help="teamspace owner, if an org (SDK default: LIGHTNING_ORG)"
    )
    common.add_argument(
        "--remote-dir", default=DEFAULT_REMOTE_DIR, help="relative to the Studio root"
    )
    common.add_argument("--archive-dir", default="scratch", help="where downloads land")

    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("launch", parents=[common], help="start or resume the sweep")
    run.add_argument(
        "--machine", default="L4", help=f"SDK machine name, e.g. {', '.join(MACHINE_HINTS)}"
    )
    run.add_argument(
        "--max-hours", type=float, default=None, help="machine runtime cap, if required"
    )
    run.add_argument("--workers", type=int, default=8, help="remote CYP_WORKERS")
    run.add_argument(
        "--members",
        type=int,
        default=1,
        help="remote CYP_MEMBERS (1 screens architectures fast; 0 uses each family's default)",
    )
    run.add_argument("--ref", default="HEAD", help="git ref to archive and run")
    run.add_argument(
        "--rev", default=None, help="override the CYP_REV namespace (default: ref SHA)"
    )
    run.add_argument(
        "--remote-python",
        default=DEFAULT_REMOTE_PYTHON,
        help="remote interpreter; the default is the env this script builds",
    )
    run.add_argument("--data-external", action="store_true", help="also ship data/external caches")
    run.add_argument("--refresh-data", action="store_true", help="resend data even if synced")
    run.add_argument(
        "--no-setup", dest="setup", action="store_false", help="skip setup.sh (env already built)"
    )
    run.add_argument(
        "--wandb",
        action="store_true",
        help="follow the sweep online: set CYP_WANDB and forward WANDB_API_KEY/PROJECT/ENTITY",
    )
    run.add_argument("--wait", action="store_true", help="block until done, then fetch")
    run.add_argument("--poll", type=int, default=300, help="seconds between checks with --wait")
    run.add_argument("--stop-after", action="store_true", help="with --wait, stop the Studio after")
    run.add_argument("--archive", default="scratch/code.tar.gz", help="local code tarball")
    run.add_argument("--data-archive", default="scratch/data.tar.gz", help="local data tarball")
    run.add_argument("--dry-run", action="store_true", help="print the plan; no cloud, no SDK")

    commands.add_parser("status", parents=[common], help="Studio and sweep state")
    commands.add_parser("fetch", parents=[common], help="download and merge results")
    halt = commands.add_parser("stop", parents=[common], help="stop the Studio (keeps its files)")
    halt.add_argument("--force", action="store_true", help="stop even if a sweep is running")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_local_env()  # before the parser: some defaults (CYP_LAI_*) are read from the environment
    args = build_parser().parse_args(argv)
    if args.command == "launch":
        if args.dry_run:
            print("Lightning AI sweep launch plan:")
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
