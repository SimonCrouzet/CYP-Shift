"""The best regression model: its members, their weights and its calibration, kept in
``configs/best_regression_model.json``.

A member is a recipe (the dotted settings of an ``ExperimentConfig``, the ones a cell of
``scripts/night_run.py`` sets) refit on all the training data with its own ensemble of members. The
model is the weighted mean of the members' predictions per endpoint, with each endpoint then
standardised to the calibration targets.

    python scripts/best_model.py check
    python scripts/best_model.py recipe MEMBER OUTPUT
    python scripts/best_model.py build [--spread F] [--output PATH]

``check`` loads every member's settings into an ``ExperimentConfig`` and validates them. ``recipe``
writes the ``focus_best.json`` that ``night_run.py``'s ``submit`` plan reads, so a member is refit
with ``modal_run.py launch --plan submit --members 0`` (put the file on the volume as
``/data/focus_best.json`` first) or locally with ``CYP_PLAN=submit``; the refit writes its
submission file, which the spec names as the member's ``file``. ``build`` blends those files by the
spec's weights, standardises the blend to the calibration targets and writes the candidate beside a
provenance file (weights, member file hashes, targets, git revision). Nothing is uploaded.
"""

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import calibration_probe
import numpy as np
import pandas as pd

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.experiment import ExperimentConfig

logger = logging.getLogger("best_model")
SPEC = Path("configs/best_regression_model.json")
KEYS = ["SMILES", "Molecule_Name"]


def load_spec(path: Path = SPEC) -> dict:
    spec = json.loads(Path(path).read_text())
    weights = [member["weight"] for member in spec["members"].values()]
    if min(weights) < 0 or not np.isclose(sum(weights), 1.0):
        raise ValueError(f"the member weights must be non-negative and sum to 1; got {weights}")
    return spec


def check(spec: dict) -> None:
    """Every member's recipe loads into an ExperimentConfig and validates."""
    for name, member in spec["members"].items():
        config = ExperimentConfig()
        for key, value in member["recipe"].items():
            config.override(key, value)
        config.validate()
        logger.info("%s: %d settings, valid", name, len(member["recipe"]))


def recipe(spec: dict, member: str) -> dict:
    """The ``focus_best.json`` content of a member: its name and its dotted settings."""
    return {"name": member, **spec["members"][member]["recipe"]}


def blend(spec: dict) -> pd.DataFrame:
    """The weighted mean of the members' submission files, endpoint by endpoint."""
    frames = {name: pd.read_csv(member["file"]) for name, member in spec["members"].items()}
    first = next(iter(frames.values()))
    columns = list(REGRESSION_ENDPOINTS)
    out = first[KEYS].copy()
    for name, frame in frames.items():
        if not frame[KEYS].equals(first[KEYS]):
            raise ValueError(f"member {name} lists different compounds, or in another order")
        if frame[columns].isna().any().any():
            raise ValueError(f"member {name} has missing predictions")
    for column in columns:
        out[column] = sum(
            spec["members"][name]["weight"] * frame[column].to_numpy(float)
            for name, frame in frames.items()
        )
    return out


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(spec: dict, output: Path, spread: float | None = None) -> Path:
    """Blend, standardise to the calibration targets and write the candidate and its provenance."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    blended = blend(spec)
    uncalibrated = output.with_name(output.stem + "_uncalibrated.csv")
    blended.to_csv(uncalibrated, index=False)
    calibration = spec["calibration"]
    spread = calibration.get("spread", 1.0) if spread is None else spread
    targets = json.loads(Path(calibration["targets"]).read_text())
    calibration_probe.apply_targets(uncalibrated, targets, output, None, spread)
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    provenance = {
        "model": spec["name"],
        "git_revision": revision,
        "spread": spread,
        "targets": calibration["targets"],
        "members": {
            name: {
                "weight": member["weight"],
                "file": member["file"],
                "sha256": _sha256(member["file"]),
            }
            for name, member in spec["members"].items()
        },
    }
    output.with_name(output.stem + ".provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("wrote %s (blend before calibration: %s)", output, uncalibrated)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", type=Path, default=SPEC)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate every member's settings.")
    recipe_parser = sub.add_parser("recipe", help="Write a member's focus_best.json.")
    recipe_parser.add_argument("member")
    recipe_parser.add_argument("output", type=Path)
    build_parser = sub.add_parser("build", help="Blend the members' files and calibrate.")
    build_parser.add_argument("--output", type=Path)
    build_parser.add_argument("--spread", type=float, help="Factor on every target's spread.")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if args.command == "check":
        check(spec)
    elif args.command == "recipe":
        args.output.write_text(json.dumps(recipe(spec, args.member), indent=2, sort_keys=True))
        logger.info("wrote %s", args.output)
    else:
        build(spec, args.output or Path(spec["output"]), args.spread)


if __name__ == "__main__":
    main()
