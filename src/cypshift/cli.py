"""Command-line entry points."""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.data.assembly import build_observations
from cypshift.data.challenge import DEFAULT_DATA_DIR, fetch_all, load
from cypshift.data.screen import pseudo_labels
from cypshift.data.splits import similarity_matched_holdout
from cypshift.decision.band import BandWidthModel
from cypshift.decision.calibration import ConformalCalibrator
from cypshift.decision.placement import place
from cypshift.decision.selection import PlacementSelector
from cypshift.eval.scoring import paired_difference, score_submission
from cypshift.eval.uncertainty import summarise
from cypshift.models.conditioned import ConditionedRoute
from cypshift.models.ensemble import DiverseEnsembleRoute
from cypshift.models.ligand import LigandOnlyRoute

app = typer.Typer(add_completion=False, help="CYP-Shift: CYP inhibition prediction.")
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def fetch(
    directory: Path = DEFAULT_DATA_DIR,
    force: Annotated[bool, typer.Option(help="Re-download even if verified.")] = False,
    verbose: bool = True,
) -> None:
    """Download the five challenge datasets and record their provenance."""
    _configure_logging(verbose)
    paths = fetch_all(directory, force)
    for name, path in paths.items():
        print(f"{name:12s} {path}")


@app.command()
def score(
    predictions: Annotated[Path, typer.Argument(help="CSV or Parquet of predictions.")],
    truth: Annotated[Path | None, typer.Option(help="Ground truth CSV.")] = None,
    directory: Path = DEFAULT_DATA_DIR,
    verbose: bool = False,
) -> None:
    """Score a regression submission with the official metric."""
    _configure_logging(verbose)
    reader = pd.read_parquet if predictions.suffix == ".parquet" else pd.read_csv
    truth_frame = pd.read_csv(truth) if truth else load("inhibition", directory)
    scores = score_submission(reader(predictions), truth_frame)
    print(scores.round(4).to_string())


@app.command()
def reference(
    directory: Path = DEFAULT_DATA_DIR,
    seed: int = 0,
    verbose: bool = False,
) -> None:
    """Print what a given prediction accuracy is worth in ST-RAE.

    Calibrates the metric against the training-set credible intervals by scoring the
    truth plus Gaussian noise of a known scale. Useful for reading a leaderboard
    position as an accuracy requirement.
    """
    _configure_logging(verbose)
    inhibition = load("inhibition", directory)
    rng = np.random.default_rng(seed)
    rows = []
    for noise in (0.0, 0.25, 0.5, 0.75, 1.0):
        submission = inhibition[["Molecule_Name"]].copy()
        for endpoint in REGRESSION_ENDPOINTS:
            submission[endpoint] = inhibition[endpoint] + rng.normal(0, noise, len(inhibition))
        scores = score_submission(submission, inhibition)["ST-RAE_mean"]
        rows.append({"noise_log_units": noise, **scores.round(3).to_dict()})
    table = pd.DataFrame(rows).set_index("noise_log_units")
    table = table[[MACRO_LABEL, *REGRESSION_ENDPOINTS]]
    table.columns = [c.split("_")[0] for c in table.columns]
    print(table.to_string())


@app.command()
def baseline(
    directory: Path = DEFAULT_DATA_DIR,
    holdout_fraction: float = 0.2,
    n_members: int = 5,
    n_estimators: int = 300,
    route_name: Annotated[
        str, typer.Option("--route", help="ligand, diverse or conditioned.")
    ] = "ligand",
    with_screen: Annotated[bool, typer.Option(help="Add screen pseudo-labels.")] = False,
    seed: int = 0,
    mask_path: Annotated[Path | None, typer.Option(help="Cache for the holdout mask.")] = None,
    verbose: bool = True,
) -> None:
    """Fit the ligand-only route and score it on the similarity-matched holdout.

    This is the floor every other route has to beat, measured through the official
    metric on a split built to resemble the blinded test set.
    """
    _configure_logging(verbose)
    train = load("inhibition", directory)
    test = load("test", directory)

    if mask_path and mask_path.exists():
        mask = np.load(mask_path)
        logger.info("reusing cached holdout of %d compounds", mask.sum())
    else:
        mask = similarity_matched_holdout(
            train["SMILES"].tolist(),
            test["SMILES"].tolist(),
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        if mask_path:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, mask)

    # Conformal calibration needs data exchangeable with what it will be applied to,
    # so the matched holdout is split at random into a calibration and an evaluation
    # half. Both then carry the same similarity profile.
    held_index = np.flatnonzero(mask)
    shuffled = np.random.default_rng(seed).permutation(held_index)
    calibration_index, evaluation_index = np.array_split(shuffled, 2)

    retained = train.loc[~mask]
    calibration = train.iloc[calibration_index]
    evaluation = train.iloc[evaluation_index]
    logger.info(
        "fit %d, calibrate %d, evaluate %d", len(retained), len(calibration), len(evaluation)
    )

    if route_name == "conditioned":
        screen_labels = None
        if with_screen:
            screen = load("screen", directory)
            kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
            screen_labels, _ = pseudo_labels(kept, retained)
        observations = build_observations(retained, screen_labels=screen_labels)
        route = ConditionedRoute(seed=seed)
        logger.info("route: %s", route.name)
        route.fit(observations)
    else:
        route = (
            DiverseEnsembleRoute(seed=seed)
            if route_name == "diverse"
            else LigandOnlyRoute(n_members=n_members, n_estimators=n_estimators, seed=seed)
        )
        logger.info("route: %s", route.name)
        route.fit(retained["SMILES"].tolist(), retained[list(REGRESSION_ENDPOINTS)].to_numpy(float))

    calibrator = ConformalCalibrator.fit(
        route.predict(calibration["SMILES"].tolist()),
        calibration[list(REGRESSION_ENDPOINTS)].to_numpy(float),
    )

    raw = route.predict(evaluation["SMILES"].tolist())
    calibrated = calibrator.apply(raw)
    calibration_prediction = calibrator.apply(route.predict(calibration["SMILES"].tolist()))

    band_models = {
        endpoint: BandWidthModel.fit(
            retained[endpoint].to_numpy(float),
            retained[f"{endpoint}_conf_low"].to_numpy(float),
            retained[f"{endpoint}_conf_high"].to_numpy(float),
        )
        for endpoint in REGRESSION_ENDPOINTS
    }

    for label, distribution in (("raw", raw), ("conformally calibrated", calibrated)):
        diagnostics = pd.DataFrame(
            {
                endpoint.split("_")[0]: summarise(
                    evaluation[endpoint].to_numpy(float),
                    distribution.mean[:, column],
                    distribution.total_std[:, column],
                )
                for column, endpoint in enumerate(REGRESSION_ENDPOINTS)
            }
        )
        print(f"\n=== predictive distribution, {label} ===")
        print(diagnostics.round(3).to_string())

    # Whether placement helps is decided on the calibration half and applied to the
    # evaluation half, never chosen on the compounds it is then scored against.
    selector = PlacementSelector.fit(
        place(calibration_prediction, band_models, seed=seed),
        calibration_prediction.mean,
        calibration,
    )
    placed_calibrated = place(calibrated, band_models, seed=seed)

    candidates = {
        "posterior mean": raw.mean,
        "placement, raw spread": place(raw, band_models, seed=seed),
        "placement, calibrated spread": placed_calibrated,
        "placement, selected endpoints": selector.apply(placed_calibrated, raw.mean),
    }
    submissions = {}
    for label, values in candidates.items():
        submission = evaluation[["Molecule_Name"]].copy()
        for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
            submission[endpoint] = values[:, column]
        submissions[label] = submission

    summary = {
        label: score_submission(frame, evaluation)["ST-RAE_mean"]
        for label, frame in submissions.items()
    }
    print("\n=== ST-RAE (lower is better) ===")
    print(pd.DataFrame(summary).round(4).to_string())

    # Differences this small are only interpretable paired: both submissions are
    # scored on identical bootstrap draws, so the difference is taken within a sample.
    reference = "posterior mean"
    for label in candidates:
        if label == reference:
            continue
        comparison = paired_difference(submissions[label], submissions[reference], evaluation)
        print(f"\n=== {label} minus {reference}, paired ===")
        print(comparison.round(4).to_string())


if __name__ == "__main__":
    app()
