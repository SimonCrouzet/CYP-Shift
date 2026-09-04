"""Command-line entry points."""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.data.challenge import DEFAULT_DATA_DIR, fetch_all, load
from cypshift.data.splits import similarity_matched_holdout
from cypshift.eval.scoring import score_submission
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
    holdout_fraction: float = 0.1,
    n_members: int = 5,
    n_estimators: int = 300,
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

    retained, held_out = train.loc[~mask], train.loc[mask]
    route = LigandOnlyRoute(n_members=n_members, n_estimators=n_estimators, seed=seed)
    route.fit(retained["SMILES"].tolist(), retained[list(REGRESSION_ENDPOINTS)].to_numpy(float))

    prediction = route.predict(held_out["SMILES"].tolist())
    submission = held_out[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = prediction.mean[:, column]

    print(score_submission(submission, held_out).round(4).to_string())


if __name__ == "__main__":
    app()
