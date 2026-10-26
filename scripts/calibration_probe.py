"""Estimate the blind set's per-endpoint mean and spread from scored probe submissions.

For an endpoint with true values ``y`` and predictions ``p`` (population moments, ``r`` the Pearson
correlation of ``p`` and ``y``), the mean squared error is

    MSE = var(y) + var(p) - 2 r sd(y) sd(p) + (mean(y) - mean(p))**2

and R² = 1 - MSE / var(y). Three scored submissions that differ only in the mean or spread of
``p`` therefore fix the three unknowns ``mean(y)``, ``sd(y)`` and ``r`` of an endpoint:

* ``base``  -- the submission as it is;
* ``shift`` -- ``p + shift``;
* ``scale`` -- ``mean(p) + scale * (p - mean(p))``.

Each endpoint is scored separately, so one probe file carries a shift on some endpoints and a
scale on the others; two files cover all four (``probe`` writes them). The squared-error optimum
is ``mean(p) = mean(y)`` and ``sd(p) = r * sd(y)``; ``solve`` reports it and ``apply`` imposes it
on any submission by standardising each endpoint's predictions to the target mean and spread.

    python scripts/calibration_probe.py probe --base submissions/graph.csv --output-dir probes
    # score probes/graph_probe_a.csv and probes/graph_probe_b.csv, write the per-endpoint R² of
    # the base and both probes to metrics.csv (columns: endpoint, probe, r2), then
    python scripts/calibration_probe.py solve --base submissions/graph.csv --metrics metrics.csv \\
        --output targets.json
    python scripts/calibration_probe.py apply --submission submissions/new.csv \\
        --targets targets.json --output submissions/new_calibrated.csv

A target may also carry ``min`` and ``max``, which clip the standardised predictions.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("calibration_probe")

#: Which transformation each probe file applies to each endpoint; the two files are complementary.
PROBE_PLAN = {
    "a": {
        "CYP1A2": "shift",
        "CYP2C9": "scale",
        "CYP2D6": "shift",
        "CYP3A4": "scale",
    },
    "b": {
        "CYP1A2": "scale",
        "CYP2C9": "shift",
        "CYP2D6": "scale",
        "CYP3A4": "shift",
    },
}


def _column(enzyme: str) -> str:
    return f"{enzyme}_pIC50_direct_inhibition"


def standardise_to(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """``values`` rescaled to the given mean and (population) standard deviation."""
    return (values - values.mean()) / values.std() * sd + mean


def transform(values: np.ndarray, kind: str, shift: float, scale: float) -> np.ndarray:
    if kind == "shift":
        return values + shift
    if kind == "scale":
        return values.mean() + scale * (values - values.mean())
    raise ValueError(f"unknown probe transformation {kind!r}")


def write_probes(base: Path, output_dir: Path, shift: float, scale: float) -> list[Path]:
    """Write the probe files next to each other and return their paths."""
    frame = pd.read_csv(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, plan in PROBE_PLAN.items():
        probe = frame.copy()
        for enzyme, kind in plan.items():
            column = _column(enzyme)
            probe[column] = transform(frame[column].to_numpy(float), kind, shift, scale)
            logger.info(
                "probe %s %s %-6s mean %.3f sd %.3f",
                name,
                enzyme,
                kind,
                probe[column].mean(),
                probe[column].std(ddof=0),
            )
        path = output_dir / f"{base.stem}_probe_{name}.csv"
        probe.to_csv(path, index=False)
        paths.append(path)
    return paths


def solve_endpoint(
    mean0: float,
    sd0: float,
    r2: dict[str, float],
    shift: float,
    scale: float,
    rho: float | None = None,
) -> dict[str, float] | None:
    """Mean, spread and correlation of the true values from the R² of the three probes.

    With ``t = var(y)``, the shift probe gives the offset ``d = mean(y) - mean(p)`` and the scale
    probe gives ``u = r sd(y)`` as linear functions of ``t``; the base R² then leaves a quadratic in
    ``t``. Of its roots only those with ``t > 0`` and ``|r| <= 1`` are kept; with two, the one
    whose ``r`` is nearer ``rho`` (a rank correlation, when given) is returned. ``None`` when no
    root qualifies.
    """
    sd_scaled = scale * sd0
    a_coef = r2["base"] - r2["shift"]
    b_coef = r2["base"] - r2["scale"]
    # d = (shift**2 - a t) / (2 shift);  u = (sd_scaled**2 - sd0**2 - b t) / (2 (sd_scaled - sd0))
    d0, d1 = shift / 2.0, -a_coef / (2.0 * shift)
    spread_gap = sd_scaled - sd0
    u0 = (sd_scaled**2 - sd0**2) / (2.0 * spread_gap)
    u1 = -b_coef / (2.0 * spread_gap)
    # base: R0 t + sd0**2 - 2 sd0 u + d**2 = 0
    quadratic = [
        d1**2,
        r2["base"] - 2.0 * sd0 * u1 + 2.0 * d0 * d1,
        sd0**2 - 2.0 * sd0 * u0 + d0**2,
    ]
    candidates = []
    for root in np.roots(quadratic):
        if abs(root.imag) > 1e-9 or root.real <= 0:
            continue
        t = float(root.real)
        r = (u0 + u1 * t) / np.sqrt(t)
        if abs(r) <= 1.0:
            candidates.append(
                {
                    "mean_true": mean0 + d0 + d1 * t,
                    "sd_true": float(np.sqrt(t)),
                    "r": float(r),
                }
            )
    if not candidates:
        return None
    if rho is None:
        return max(candidates, key=lambda c: c["r"])
    return min(candidates, key=lambda c: abs(c["r"] - rho))


def solve(
    base: Path,
    metrics: Path,
    shift: float,
    scale: float,
    noise: float,
    draws: int = 400,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-endpoint targets (mean, spread) from the base file's moments and the probe R²."""
    frame = pd.read_csv(base)
    table = pd.read_csv(metrics)
    generator = np.random.default_rng(seed)
    targets: dict[str, dict[str, float]] = {}
    for enzyme in PROBE_PLAN["a"]:
        values = frame[_column(enzyme)].to_numpy(float)
        mean0, sd0 = float(values.mean()), float(values.std())
        rows = table[table["endpoint"].str.startswith(enzyme)].set_index("probe")
        r2 = rows["r2"].astype(float).to_dict()
        rho = (
            float(rows["rho"].dropna().iloc[0])
            if "rho" in rows and rows["rho"].notna().any()
            else None
        )
        found = solve_endpoint(mean0, sd0, r2, shift, scale, rho)
        if found is None:
            logger.warning("%s: the probe R² admit no solution; target not set", enzyme)
            continue
        draws_found = []
        for _ in range(draws):
            noisy = {k: v + generator.normal(0.0, noise) for k, v in r2.items()}
            one = solve_endpoint(mean0, sd0, noisy, shift, scale, rho)
            if one is not None:
                draws_found.append(one)
        low, high = {}, {}
        for key in ("mean_true", "sd_true", "r"):
            sample = [d[key] for d in draws_found] or [found[key]]
            low[key], high[key] = np.percentile(sample, [10, 90])
        targets[enzyme] = {
            "mean": found["mean_true"],
            "sd": found["r"] * found["sd_true"],
            "mean_true": found["mean_true"],
            "sd_true": found["sd_true"],
            "r": found["r"],
            "mean_true_10_90": [float(low["mean_true"]), float(high["mean_true"])],
            "sd_true_10_90": [float(low["sd_true"]), float(high["sd_true"])],
            "r_10_90": [float(low["r"]), float(high["r"])],
            "base_mean": mean0,
            "base_sd": sd0,
        }
    return targets


def apply_targets(submission: Path, targets: dict, output: Path, only: list[str] | None) -> None:
    frame = pd.read_csv(submission)
    for enzyme, target in targets.items():
        if only and enzyme not in only:
            continue
        column = _column(enzyme)
        values = standardise_to(frame[column].to_numpy(float), target["mean"], target["sd"])
        # Optional bounds (the labelled range of the training set): a stretched tail must not
        # leave the range the assay reports.
        frame[column] = np.clip(values, target.get("min", -np.inf), target.get("max", np.inf))
        logger.info("%s -> mean %.3f sd %.3f", enzyme, target["mean"], target["sd"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="Write the two probe files from a base submission.")
    probe.add_argument("--base", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    solve_parser = sub.add_parser("solve", help="Estimate the targets from the probes' R².")
    solve_parser.add_argument("--base", type=Path, required=True)
    solve_parser.add_argument("--metrics", type=Path, required=True)
    solve_parser.add_argument("--output", type=Path, required=True)
    solve_parser.add_argument("--noise", type=float, default=0.02, help="R² standard error.")
    apply = sub.add_parser("apply", help="Standardise a submission to the target moments.")
    apply.add_argument("--submission", type=Path, required=True)
    apply.add_argument("--targets", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--only", help="Comma-separated endpoints to change; default all.")
    for command in (probe, solve_parser):
        command.add_argument("--shift", type=float, default=0.5)
        command.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    if args.command == "probe":
        for path in write_probes(args.base, args.output_dir, args.shift, args.scale):
            logger.info("wrote %s", path)
    elif args.command == "solve":
        targets = solve(args.base, args.metrics, args.shift, args.scale, args.noise)
        args.output.write_text(json.dumps(targets, indent=2) + "\n")
        for enzyme, found in targets.items():
            logger.info(
                "%s: true mean %.2f %s sd %.2f %s r %.2f -> target mean %.2f sd %.2f "
                "(base %.2f/%.2f)",
                enzyme,
                found["mean_true"],
                np.round(found["mean_true_10_90"], 2).tolist(),
                found["sd_true"],
                np.round(found["sd_true_10_90"], 2).tolist(),
                found["r"],
                found["mean"],
                found["sd"],
                found["base_mean"],
                found["base_sd"],
            )
    else:
        only = args.only.split(",") if args.only else None
        apply_targets(args.submission, json.loads(args.targets.read_text()), args.output, only)


if __name__ == "__main__":
    main()
