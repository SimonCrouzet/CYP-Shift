"""Launch the night_run sweep on a Lightning AI Studio GPU and sync the results back.

The sweep (``scripts/night_run.py``) is GPU-bound and env-var driven (``CYP_WORKERS`` /
``CYP_MEMBERS``), writing its results under ``data/`` and ``submissions/``. This script runs
that same sweep on a Lightning AI Studio instead of the local machine and pulls the outputs
home, so a RAM/GPU-constrained laptop does not have to host the run.

Flow (each step is a single, stable Lightning SDK call, so the cloud surface stays small):

    1. ``git archive`` the tracked tree -> a code tarball (no data, no .git, no working configs).
    2. ``tar`` the data the sweep needs (``data/raw`` core, optionally the ``data/external``
       auxiliary caches) -> a data tarball.
    3. Start the Studio on the requested machine.
    4. Upload both tarballs; extract them into the remote working directory.
    5. Optionally run ``./setup.sh`` there (build the env, fetch/verify the core data), then run
       the sweep with the chosen ``CYP_WORKERS`` / ``CYP_MEMBERS``.
    6. ``tar`` the result paths that exist on the remote, download that one archive, extract it
       locally (into the repo, so bundles and the results CSV land where the tooling expects).
    7. Stop the Studio (unless ``--keep-running``).

Auth is the Lightning SDK's own: ``LIGHTNING_USER_ID`` + ``LIGHTNING_API_KEY`` in the
environment, or a prior ``lightning login``. Nothing here reads or stores a credential.

``--dry-run`` prints the full plan -- every command and transfer -- and touches neither the
cloud nor the SDK, so the orchestration can be checked without an account.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("lightning_run")

#: The sweep's outputs, relative to the repo root. Only those that exist on the remote are
#: archived for download (a members-screening run writes the same paths as a full run).
RESULT_PATHS = (
    "data/night_results.csv",
    "data/run_bundles",
    "configs/night_best.json",
    "configs/night_best_tdi.json",
    "submissions/night_submission_regression.csv",
    "submissions/night_submission_classification.csv",
)

#: Machine names accepted by ``--machine``, mapped to the Lightning SDK ``Machine`` attribute.
#: Kept as strings so the mapping is inspectable (and testable) without importing the SDK.
MACHINES = {
    "CPU": "CPU",
    "T4": "T4",
    "L4": "L4",
    "A10G": "A10G",
    "L40S": "L40S",
    "A100": "A100",
}


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

    ``data/raw`` holds the challenge files the run cannot fetch itself in-process; ``data/external``
    holds the auxiliary caches (octant/tox21/veith/chembl/sair/novartis, embeddings, the CheMeleon
    encoder). Some of those download on first use remotely, but the local-only caches (e.g. octant,
    novartis) must be shipped, so uploading ``data/external`` is offered when it exists.
    """
    sources = ["data/raw"]
    if include_external:
        sources.append("data/external")
    return sources


def wandb_env(enabled: bool) -> dict[str, str]:
    """The W&B environment to forward to the remote sweep when ``--wandb`` is on.

    Turns the sweep's ``CYP_WANDB`` gate on and carries whichever standard W&B variables are set
    locally (the API key for non-interactive auth, the project/entity for where the run lands), so
    the remote run reports to the same account and project without an interactive ``wandb login``.
    """
    import os

    if not enabled:
        return {}
    env = {"CYP_WANDB": "1"}
    for name in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def sweep_command(
    remote_dir: str,
    python: str,
    workers: int,
    members: int,
    run_setup: bool,
    extra_env: dict[str, str] | None = None,
) -> str:
    """The remote shell command that (optionally) sets up the env and runs the sweep.

    ``CYP_WORKERS`` / ``CYP_MEMBERS`` are passed exactly as a local launch would set them;
    ``python`` is the interpreter (or interpreter prefix, e.g. ``conda run -n cypshift python``)
    that has the dependencies -- after ``./setup.sh`` on a fresh Studio, or the Studio's own env.
    ``extra_env`` carries any W&B variables so the remote run can be followed online.
    """
    setup = "./setup.sh && " if run_setup else ""
    env = {"CYP_WORKERS": str(workers), "CYP_MEMBERS": str(members), **(extra_env or {})}
    assignments = " ".join(f"{key}={value}" for key, value in env.items())
    return f"cd {remote_dir} && {setup}{assignments} {python} scripts/night_run.py"


def result_tar_command(remote_dir: str, archive_name: str) -> str:
    """A remote command that tars only the result paths that exist, into ``archive_name``.

    The run may not write every path (an empty TDI track leaves no TDI submission), so each is
    added only ``if`` present -- ``tar`` on a missing path would fail the whole download.
    """
    existing = " ".join(f"$(test -e {path} && echo {path})" for path in RESULT_PATHS)
    return f"cd {remote_dir} && tar czf {archive_name} {existing}"


def resolve_machine(name: str) -> object:
    """Map a ``--machine`` string to a Lightning SDK ``Machine`` member (SDK imported lazily)."""
    key = name.upper()
    if key not in MACHINES:
        raise SystemExit(f"unknown --machine {name!r}; choose one of {sorted(MACHINES)}")
    machine_module = _require_sdk()[1]
    return getattr(machine_module, MACHINES[key])


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


def plan(args: argparse.Namespace) -> list[str]:
    """The ordered, human-readable steps the launch will take -- the body of ``--dry-run``.

    Pure: it builds the plan from the arguments alone, with no cloud call and no SDK import, so it
    is the testable description of exactly what ``launch`` does.
    """
    code = f"{args.remote_dir}/code.tar.gz"
    data = f"{args.remote_dir}/data.tar.gz"
    results = f"{args.remote_dir}/results.tar.gz"
    steps = [
        f"git archive {args.ref} -> {args.archive}",
        f"tar {' '.join(data_tar_sources(args.data_external))} -> {args.data_archive}",
        f"Studio(name={args.studio!r}, teamspace={args.teamspace!r}).start({args.machine})",
        f"upload {args.archive} -> {code}; upload {args.data_archive} -> {data}",
        f"remote: mkdir -p {args.remote_dir} && tar xzf {code} -C {args.remote_dir} && "
        f"tar xzf {data} -C {args.remote_dir}",
        "remote: "
        + sweep_command(
            args.remote_dir,
            args.remote_python,
            args.workers,
            args.members,
            args.setup,
            # Never print the API key in the plan; show it is forwarded, not its value.
            {k: ("***" if k == "WANDB_API_KEY" else v) for k, v in wandb_env(args.wandb).items()},
        ),
        "remote: " + result_tar_command(args.remote_dir, "results.tar.gz"),
        f"download {results} -> {args.archive_dir}/results.tar.gz; extract into the repo",
        "Studio.stop()" if not args.keep_running else "Studio left running (--keep-running)",
    ]
    return steps


def launch(args: argparse.Namespace) -> None:
    """Run the plan against a real Studio. The SDK is imported here, never at module load."""
    studio_cls, _ = _require_sdk()
    machine = resolve_machine(args.machine)

    archive = build_code_archive(Path(args.archive), args.ref)
    data_archive = Path(args.data_archive)
    data_archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "czf", str(data_archive), *data_tar_sources(args.data_external)], check=True
    )

    studio = studio_cls(name=args.studio, teamspace=args.teamspace, create_ok=True)
    logger.info("starting Studio %s on %s", args.studio, args.machine)
    studio.start(machine)
    try:
        remote_code = f"{args.remote_dir}/code.tar.gz"
        remote_data = f"{args.remote_dir}/data.tar.gz"
        studio.run(f"mkdir -p {args.remote_dir}")
        studio.upload_file(str(archive), remote_code)
        studio.upload_file(str(data_archive), remote_data)
        studio.run(f"cd {args.remote_dir} && tar xzf code.tar.gz && tar xzf data.tar.gz")
        logger.info("running the sweep (workers=%d members=%d)", args.workers, args.members)
        output = studio.run(
            sweep_command(
                args.remote_dir,
                args.remote_python,
                args.workers,
                args.members,
                args.setup,
                wandb_env(args.wandb),
            )
        )
        logger.info("sweep finished:\n%s", output)

        studio.run(result_tar_command(args.remote_dir, "results.tar.gz"))
        local_results = Path(args.archive_dir) / "results.tar.gz"
        local_results.parent.mkdir(parents=True, exist_ok=True)
        studio.download_file(f"{args.remote_dir}/results.tar.gz", str(local_results))
        subprocess.run(["tar", "xzf", str(local_results)], check=True)
        logger.info("results extracted into the repo (%s)", ", ".join(RESULT_PATHS))
    finally:
        if not args.keep_running:
            logger.info("stopping Studio %s", args.studio)
            studio.stop()


def build_parser() -> argparse.ArgumentParser:
    import os

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--studio", default=os.environ.get("CYP_LAI_STUDIO", "cyp-shift"))
    parser.add_argument("--teamspace", default=os.environ.get("CYP_LAI_TEAMSPACE"))
    parser.add_argument("--machine", default="A10G", help=f"one of {sorted(MACHINES)}")
    parser.add_argument("--workers", type=int, default=8, help="remote CYP_WORKERS")
    parser.add_argument(
        "--members",
        type=int,
        default=1,
        help="remote CYP_MEMBERS (1 screens architectures fast; 0 uses each family's default)",
    )
    parser.add_argument("--ref", default="HEAD", help="git ref to archive and run")
    parser.add_argument("--remote-dir", default="~/cyp-shift", help="remote working directory")
    parser.add_argument(
        "--remote-python",
        default="python",
        help="remote interpreter (or prefix, e.g. 'conda run -n cypshift python')",
    )
    parser.add_argument(
        "--data-external",
        action="store_true",
        help="also ship data/external (local-only auxiliary caches); else fetch remotely",
    )
    parser.add_argument(
        "--no-setup",
        dest="setup",
        action="store_false",
        help="skip ./setup.sh on the remote (the Studio already has the env and data)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="follow the sweep online: set CYP_WANDB on the remote and forward WANDB_API_KEY / "
        "WANDB_PROJECT / WANDB_ENTITY from this environment",
    )
    parser.add_argument("--keep-running", action="store_true", help="do not stop the Studio after")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no cloud, no SDK")
    parser.add_argument("--archive", default="scratch/code.tar.gz", help="local code tarball path")
    parser.add_argument(
        "--data-archive", default="scratch/data.tar.gz", help="local data tarball path"
    )
    parser.add_argument(
        "--archive-dir", default="scratch", help="where the downloaded results archive lands"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print("Lightning AI sweep launch plan:")
        for index, step in enumerate(plan(args), start=1):
            print(f"  {index}. {step}")
        return
    launch(args)


if __name__ == "__main__":
    sys.exit(main())
