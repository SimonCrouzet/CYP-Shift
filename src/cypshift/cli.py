"""Command-line entry points."""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from cypshift.config import MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.data.challenge import DEFAULT_DATA_DIR, fetch_all, load
from cypshift.eval.scoring import score_submission

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


if __name__ == "__main__":
    app()
