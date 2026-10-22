"""Command-line entry points."""

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from cypshift.config import CLASSIFICATION_ENDPOINTS, ISOFORMS, MACRO_LABEL, REGRESSION_ENDPOINTS
from cypshift.data.assembly import build_observations
from cypshift.data.challenge import DEFAULT_DATA_DIR, fetch_all, load
from cypshift.data.screen import pseudo_labels
from cypshift.data.splits import similarity_matched_holdout
from cypshift.decision.calibration import ConformalCalibrator
from cypshift.eval.scoring import paired_difference, score_submission
from cypshift.eval.uncertainty import summarise
from cypshift.experiment import ExperimentConfig
from cypshift.models.boosted import BoostedTrees
from cypshift.models.ensemble import PerTargetEnsemble
from cypshift.models.forest import PerTargetForest
from cypshift.models.shared import SharedEnsemble
from cypshift.pipeline import assemble_training_data, build_model

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
        str, typer.Option("--model", help="ligand, diverse or conditioned.")
    ] = "ligand",
    with_screen: Annotated[bool, typer.Option(help="Add screen pseudo-labels.")] = False,
    seed: int = 0,
    mask_path: Annotated[Path | None, typer.Option(help="Cache for the holdout mask.")] = None,
    verbose: bool = True,
) -> None:
    """Fit the ligand-only model and score it on the similarity-matched holdout.

    This is the floor every other model has to beat, measured through the official
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
        model = SharedEnsemble(seed=seed)
        logger.info("model: %s", model.name)
        model.fit(observations)
    else:
        model = (
            PerTargetEnsemble(seed=seed)
            if route_name == "diverse"
            else PerTargetForest(n_members=n_members, n_estimators=n_estimators, seed=seed)
        )
        logger.info("model: %s", model.name)
        model.fit(retained["SMILES"].tolist(), retained[list(REGRESSION_ENDPOINTS)].to_numpy(float))

    calibrator = ConformalCalibrator.fit(
        model.predict(calibration["SMILES"].tolist()),
        calibration[list(REGRESSION_ENDPOINTS)].to_numpy(float),
    )

    raw = model.predict(evaluation["SMILES"].tolist())
    calibrated = calibrator.apply(raw)

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

    # The submission is the model's prediction; the posterior mean is that number.
    # Score it on the evaluation half through the official metric.
    submission = evaluation[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = raw.mean[:, column]
    scores = score_submission(submission, evaluation)
    print("\n=== ST-RAE (lower is better) ===")
    print(scores["ST-RAE_mean"].round(4).to_string())


@app.command()
def submit(
    output: Annotated[Path, typer.Argument(help="Where to write the submission CSV.")],
    directory: Annotated[
        Path | None,
        typer.Option(help="Data root; defaults to the config's data.directory."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Experiment config; defaults to configs/default.json."),
    ] = None,
    track: Annotated[
        str,
        typer.Option(help='"regression" or "classification"; they want different models.'),
    ] = "regression",
    verbose: bool = True,
) -> None:
    """Fit on all training data and write predictions for the blinded test set.

    Trains on everything, unlike `baseline`, which holds a similarity-matched split back
    to score against. Nothing here is scored: the blinded file carries only
    ``Molecule_Name`` and ``SMILES``, and the labels stay with the organisers.

    ``--track`` picks which leaderboard the run is for, and the two want different
    models. The classification track needs the preincubated assay_condition in training, without
    which a curve model answers "no" for every compound and scores MCC 0.0; that assay_condition
    costs about 0.04 macro ST-RAE on the regression track at every weight tried. Since
    the tracks are scored separately there is no reason to compromise -- run this twice.

    Each track writes exactly the columns its leaderboard wants, always led by
    ``SMILES`` and ``Molecule_Name``: the four pIC50 columns for regression, the two
    ``is_TDI`` booleans for classification. The classification track therefore needs a
    model that can produce TDI calls -- the label compares the preincubated assay_condition's
    midpoint against the direct assay_condition's, so a model emitting one pIC50 has nothing to
    compare and the command refuses rather than write an invalid file.
    """
    _configure_logging(verbose)
    if track not in ("regression", "classification"):
        raise typer.BadParameter('track must be "regression" or "classification"')
    config = ExperimentConfig.load(config_path) if config_path else ExperimentConfig.load()
    # Resolve a single data root: an explicit --directory wins, otherwise the config's own
    # directory is honoured rather than silently overwritten by the CLI default. The resolved
    # path is then used for both config.data.directory (which assemble_training_data reads for
    # the screen/TDI/exclusion data) and the train/test loads, so they cannot diverge.
    directory = Path(directory) if directory is not None else Path(config.data.directory)
    config.data.directory = str(directory)
    # The classification track needs the preincubated assay_condition, so require it rather than
    # silently flipping the regression track's off: the submitted recipe stays the one that
    # was validated, and a config that deliberately kept the assay_condition on for regression is
    # honoured.
    if track == "classification" and not config.sources.tdi_assay_condition:
        logger.info(
            "classification track needs the preincubated assay_condition; "
            "enabling sources.tdi_assay_condition"
        )
        config.sources.tdi_assay_condition = True
    config.validate()

    train = load("inhibition", directory)
    test = load("test", directory)
    logger.info("training on all %d compounds, predicting %d", len(train), len(test))

    # Full-data fit: only the blinded-test compounds are kept out of the TDI assay_condition.
    observations, screen_model = assemble_training_data(
        config, train, exclude_names=set(test["Molecule_Name"])
    )
    scored = list(ISOFORMS)
    present = observations.get("target", [])
    extra = [t for t in dict.fromkeys(present) if t not in scored]
    model = build_model(config, screen_model, targets=tuple(scored + sorted(extra)))
    # Check the model can serve the track before the expensive fit, not after.
    if track == "classification" and not hasattr(model, "predict_is_time_dependent"):
        raise typer.BadParameter(
            f"{model.name} cannot predict TDI; the classification track needs a model that can"
        )
    model.fit(observations)
    # Persist the exact recipe that produced the file, next to it.
    config.save(output.parent / f"{output.stem}.config.json")

    # The leaderboard keys on Molecule_Name but requires the SMILES column too, and
    # validates the exact column set per track, so each track writes only its own.
    submission = test[["SMILES", "Molecule_Name"]].copy()
    if track == "regression":
        prediction = model.predict(test["SMILES"].tolist())
        for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
            submission[endpoint] = prediction.mean[:, column]
        required = list(REGRESSION_ENDPOINTS)
    else:
        flags = model.predict_is_time_dependent(test["SMILES"].tolist())
        for endpoint in CLASSIFICATION_ENDPOINTS:
            submission[endpoint] = flags[:, list(ISOFORMS).index(endpoint.split("_")[0])]
        required = list(CLASSIFICATION_ENDPOINTS)

    values = submission[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise typer.BadParameter("predictions contain NaN or inf, which the leaderboard rejects")
    if len(submission) != len(test):
        raise typer.BadParameter(f"expected {len(test)} rows, built {len(submission)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    logger.info("wrote %d rows and %d columns to %s", len(submission), submission.shape[1], output)
    typer.echo(f"{len(submission)} predictions written to {output}")


@app.command()
def loss_ablation(
    directory: Path = DEFAULT_DATA_DIR,
    holdout_fraction: float = 0.2,
    with_screen: Annotated[bool, typer.Option(help="Add screen pseudo-labels.")] = True,
    learner: Annotated[str, typer.Option(help="boosted or neural.")] = "boosted",
    epochs: int = 60,
    seed: int = 0,
    mask_path: Annotated[Path | None, typer.Option(help="Cache for the holdout mask.")] = None,
    verbose: bool = True,
) -> None:
    """Train one architecture on each candidate objective and report every metric.

    A single-variable sweep: same network, same data, same split, same scoring, with
    only the training objective changing. Every metric is reported for every assay_condition
    regardless of which one it trained on, since a model tuned for one can easily be
    worse on another and that trade is the thing worth seeing.
    """
    _configure_logging(verbose)
    train = load("inhibition", directory)
    test = load("test", directory)

    if mask_path and mask_path.exists():
        mask = np.load(mask_path)
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

    retained, evaluation = train.loc[~mask], train.loc[mask]
    screen_labels = None
    if with_screen:
        screen = load("screen", directory)
        kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
        screen_labels, _ = pseudo_labels(kept, retained)
    observations = build_observations(retained, screen_labels=screen_labels)

    # Imported here rather than at module scope, so the package and the commands that
    # need no network work without the optional deep-learning dependencies.
    from cypshift.models.neural import OBJECTIVES, NeuralNetwork

    submissions = {}
    for objective in OBJECTIVES:
        logger.info("training with the %s objective", objective)
        model = (
            BoostedTrees(loss=objective, seed=seed)
            if learner == "boosted"
            else NeuralNetwork(loss=objective, epochs=epochs, seed=seed)
        ).fit(observations)
        predicted = model.predict(evaluation["SMILES"].tolist()).mean
        frame = evaluation[["Molecule_Name"]].copy()
        for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
            frame[endpoint] = predicted[:, column]
        submissions[objective] = frame

    scored = {label: score_submission(frame, evaluation) for label, frame in submissions.items()}
    for metric in [c for c in next(iter(scored.values())).columns if c.endswith("_mean")]:
        table = pd.DataFrame({label: frame[metric] for label, frame in scored.items()})
        print(f"\n=== {metric.removesuffix('_mean')}, by training objective ===")
        print(table.round(4).to_string())

    for objective in OBJECTIVES:
        if objective == "squared":
            continue
        comparison = paired_difference(submissions[objective], submissions["squared"], evaluation)
        print(f"\n=== ST-RAE: {objective} minus squared error, paired ===")
        print(comparison.round(4).to_string())


if __name__ == "__main__":
    app()
