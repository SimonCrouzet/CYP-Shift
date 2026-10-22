#!/usr/bin/env python
"""Preflight for a cluster sweep: interpreter, packages, GPU, data caches, tokens.

Run before ``scripts/night_run.py`` so a multi-hour sweep is not launched against a half-built
environment. Reports what each dependency and data cache unlocks, then prints which sweep config
families can run and exits non-zero when the core (challenge data + the always-required packages)
is missing. Optional packages and auxiliary caches only gate their own config families, so a
missing one is a warning, not a failure.

    /mnt/wsl/ext/envs/cypshift/bin/python scripts/check_env.py
    /mnt/wsl/ext/envs/cypshift/bin/python scripts/check_env.py --fetch   # download core data first
"""

import argparse
import importlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

# Minimum interpreter, matching pyproject's requires-python.
MIN_PYTHON = (3, 11)

# (import name, distribution name) for the packages the boosted pipeline always needs.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("rdkit", "rdkit"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("lightgbm", "lightgbm"),
    ("dimorphite_dl", "dimorphite-dl"),
    ("typer", "typer"),
    ("requests", "requests"),
    ("yaml", "pyyaml"),
]

# Optional backends, each with the config families it unlocks.
OPTIONAL_PACKAGES = [
    ("torch", "torch", "graph / neural / curve models"),
    ("chemprop", "chemprop", "graph (D-MPNN) model"),
    ("tabpfn", "tabpfn", "tabular foundation (tfm) model"),
]

# Data the challenge itself provides; without these nothing runs. Relative to the data dir.
CORE_DATA = [
    "cyp-challenge-TRAIN_inhibition.csv",
    "cyp-challenge-single-concentration-TRAIN.csv",
    "cyp-challenge-TEST-BLINDED.csv",
]
# The TDI track file; its absence disables only the TDI phase.
TDI_DATA = ["cyp-challenge-TRAIN_TDI.csv"]


def _mark(ok: bool, warn: bool = False) -> str:
    return "OK  " if ok else ("WARN" if warn else "MISS")


def check_python() -> bool:
    ok = sys.version_info[:2] >= MIN_PYTHON
    want = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, sys.version_info[:3]))
    print(f"[{_mark(ok)}] python {have} (need >= {want}) at {sys.executable}")
    return ok


def _try_import(import_name: str, dist_name: str) -> str | None:
    """Return the installed version, or ``None`` if the package cannot be imported."""
    try:
        importlib.import_module(import_name)
    except Exception:  # noqa: BLE001 -- any import failure means it is unusable here
        return None
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "?"


def check_packages() -> tuple[list[str], list[str]]:
    """Report every package; return (missing required imports, missing optional imports)."""
    missing_required, missing_optional = [], []
    print("\npackages (required):")
    for import_name, dist_name in REQUIRED_PACKAGES:
        version = _try_import(import_name, dist_name)
        print(f"  [{_mark(version is not None)}] {import_name:16} {version or '-'}")
        if version is None:
            missing_required.append(import_name)
    print("\npackages (optional backends):")
    for import_name, dist_name, unlocks in OPTIONAL_PACKAGES:
        version = _try_import(import_name, dist_name)
        mark = _mark(version is not None, warn=True)
        print(f"  [{mark}] {import_name:16} {version or '-':10} {unlocks}")
        if version is None:
            missing_optional.append(import_name)
    return missing_required, missing_optional


def check_cuda() -> bool:
    """Report CUDA availability; a CPU-only node runs the GPU families, only far slower."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("\n[WARN] torch not importable; GPU families unavailable")
        return False
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    name = torch.cuda.get_device_name(0) if available else "none"
    print(f"\n[{_mark(available, warn=True)}] CUDA available={available} devices={count} ({name})")
    if not available:
        print("       GPU families (graph/neural/curve/tfm) will run on CPU -- hours become days")
    return available


def _present(path: Path) -> bool:
    return path.exists() and (path.is_file() or any(path.iterdir()))


def check_data(data_dir: Path, external: Path) -> bool:
    """Report core and auxiliary data; return True when the core challenge files are present."""
    print(f"\ncore challenge data ({data_dir}):")
    core_ok = True
    for filename in CORE_DATA:
        present = (data_dir / filename).exists()
        print(f"  [{_mark(present)}] {filename}")
        core_ok = core_ok and present
    for filename in TDI_DATA:
        present = (data_dir / filename).exists()
        print(f"  [{_mark(present, warn=True)}] {filename} (TDI track)")

    # Each auxiliary cache gates only the config families whose name carries its source.
    aux = {
        "octant": external / "octant" / "inhibition.tsv",
        "tox21": external / "tox21",
        "veith": external / "veith",
        "chembl": external / "chembl_CYP3A4.csv",
        "sair": external / "sair.parquet",
        "novartis": external / "novartis" / "cyp3a4_v1.csv",
        "mammal embedding": external / "embeddings" / "mammal.parquet",
        "chemeleon weights": external / "chemeleon" / "chemeleon_mp.pt",
        "ionization": external / "ionization" / "features.parquet",
    }
    print(f"\nauxiliary data / caches ({external}):")
    for label, path in aux.items():
        print(f"  [{_mark(_present(path), warn=True)}] {label:18} {path}")
    return core_ok


def check_tokens() -> None:
    """Report the environment tokens some families need; all are optional."""
    print("\nenvironment tokens:")
    tfm_token = bool(os.environ.get("TABPFN_TOKEN"))
    tfm_note = "set" if tfm_token else "unset (tfm configs fail+skip)"
    print(f"  [{_mark(tfm_token, warn=True)}] TABPFN_TOKEN {tfm_note}")
    hf = bool(os.environ.get("HF_TOKEN") or os.environ.get("HF_HOME"))
    hf_note = "set" if hf else "unset (only needed to (re)generate embeddings)"
    print(f"  [{_mark(hf, warn=True)}] HF_TOKEN / HF_HOME {hf_note}")


def fetch_core(data_dir: Path) -> None:
    """Download the core challenge files into ``data_dir`` if they are missing."""
    from cypshift.data.challenge import FILENAMES, fetch

    wanted = [name for name, filename in FILENAMES.items() if filename in CORE_DATA + TDI_DATA]
    for name in wanted:
        fetch(name, directory=data_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--external", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--fetch", action="store_true", help="download missing core challenge files first"
    )
    arguments = parser.parse_args(argv)

    python_ok = check_python()
    missing_required, _ = check_packages()
    check_cuda()

    if arguments.fetch and not missing_required:
        try:
            fetch_core(arguments.data_dir)
        except Exception as error:  # noqa: BLE001 -- report, do not crash the preflight
            print(f"\n[WARN] --fetch failed: {error}")

    core_ok = check_data(arguments.data_dir, arguments.external)
    check_tokens()

    # cypshift itself must import, or none of the scripts run.
    package_ok = _try_import("cypshift", "cypshift") is not None
    print(f"\n[{_mark(package_ok)}] cypshift importable")

    ready = python_ok and not missing_required and core_ok and package_ok
    print("\n" + ("=" * 60))
    if ready:
        print("READY: core dependencies and challenge data present. Launch:")
        print("  CYP_WORKERS=8 CYP_MEMBERS=1 python scripts/night_run.py")
        print("Auxiliary/embedding families above marked MISS will fail+skip per config.")
    else:
        print("NOT READY. Fix the MISS lines above before launching:")
        if not python_ok:
            print(f"  - interpreter older than {'.'.join(map(str, MIN_PYTHON))}")
        if missing_required:
            print(f"  - missing required packages: {', '.join(missing_required)}")
        if not core_ok:
            print("  - missing core challenge data (re-run with --fetch, or sync data/raw)")
        if not package_ok:
            print("  - cypshift not importable (pip install -e .)")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
