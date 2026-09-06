"""Turning a configuration into a fitted model and a score.

One place where a configuration becomes an experiment, so a sweep and a single run
follow exactly the same path and cannot drift apart.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import CLASSIFICATION_ENDPOINTS, ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import SourceWeights, build_observations
from cypshift.data.challenge import load
from cypshift.data.screen import pseudo_labels, response_observations
from cypshift.data.splits import similarity_matched_holdout
from cypshift.data.surrogate import SURROGATE_STRATEGIES, fit_surrogate
from cypshift.eval.classification import score_classification
from cypshift.eval.scoring import score_submission
from cypshift.experiment import MULTIFIDELITY_STRATEGIES, ExperimentConfig
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)


def build_model(config: ExperimentConfig, surrogate: object | None = None) -> object:
    """Construct the model a configuration names.

    Imported lazily so a run needing only boosted trees does not pay for torch, and a
    missing optional dependency fails only when that model is actually requested.
    """
    kind = config.model.kind
    shared = dict(
        conditioning=config.conditioning.strategy,
        n_bits=config.features.fingerprint_bits,
        fingerprint_radius=config.features.fingerprint_radius,
        physchem=config.features.use_physicochemical,
        mechanism=config.features.use_mechanistic,
        seed=config.data.seed,
        surrogate=surrogate,
    )
    if kind == "boosted":
        from cypshift.models.boosted import BoostedTrees

        return BoostedTrees(
            loss=config.model.objective,
            n_members=config.model.n_members,
            n_estimators=config.model.boosted.n_estimators,
            learning_rate=config.model.boosted.learning_rate,
            num_leaves=config.model.boosted.num_leaves,
            min_child_samples=config.model.boosted.min_child_samples,
            **shared,
        )
    if kind == "neural":
        from cypshift.models.neural import NeuralNetwork

        return NeuralNetwork(
            loss=config.model.objective,
            hidden=config.model.neural.hidden,
            dropout=config.model.neural.dropout,
            epochs=config.model.neural.epochs,
            batch_size=config.model.neural.batch_size,
            learning_rate=config.model.neural.learning_rate,
            n_forward_passes=config.model.neural.forward_passes,
            **shared,
        )
    if kind == "graph":
        from cypshift.models.graph import GraphNetwork

        return GraphNetwork(
            loss=config.model.objective,
            hidden=config.model.graph.hidden,
            depth=config.model.graph.depth,
            dropout=config.model.graph.dropout,
            epochs=config.model.graph.epochs,
            batch_size=config.model.graph.batch_size,
            n_members=config.model.n_members,
            conditioning=config.conditioning.strategy,
            seed=config.data.seed,
        )
    if kind == "curve":
        from cypshift.models.curve import CurveModel

        return CurveModel(
            n_bits=config.features.fingerprint_bits,
            fingerprint_radius=config.features.fingerprint_radius,
            physchem=config.features.use_physicochemical,
            mechanism=config.features.use_mechanistic,
            hidden=config.model.curve.hidden,
            dropout=config.model.curve.dropout,
            epochs=config.model.curve.epochs,
            batch_size=config.model.curve.batch_size,
            learning_rate=config.model.curve.learning_rate,
            n_forward_passes=config.model.curve.forward_passes,
            conditioning=config.model.curve.conditioning,
            descriptor_sources=tuple(config.model.curve.descriptor_sources),
            seed=config.data.seed,
            surrogate=surrogate,
        )
    raise ValueError(f"model.kind={kind!r} has no builder")


def supported_multifidelity(kind: str) -> tuple[str, ...]:
    """Which multi-fidelity strategies a model family can actually use.

    The surrogate strategies fit a model to the screen and hand its output, or its
    hidden layer, to the real model as an input column. That is a way of using a cheap
    measurement when the model cannot consume the measurement itself, and two families
    do not need it:

    ``curve``
        The screen is a reading of inhibition at 4.95e-5 M, which is a point on the very
        curve this model predicts. It enters through ``extra_rows`` as an observation in
        its own units, with no calibration and nothing fitted in between. Passing the
        same readings through a surrogate instead would replace a direct measurement
        with a prediction of it, which is strictly worse and was the reason for
        modelling a curve in the first place.
    ``graph``
        Consumes molecular graphs rather than a dense feature matrix, so a surrogate
        column has nowhere to go.

    Refusing beats accepting the setting and quietly training without it, which would
    report a comparison that never happened.
    """
    if kind in ("curve", "graph"):
        return ("none", "extra_rows")
    return MULTIFIDELITY_STRATEGIES


def assemble_training_data(
    config: ExperimentConfig, retained: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Build the observation table the configured multi-fidelity strategy calls for.

    Returns the observations and, where the strategy produces one, a per-compound
    surrogate feature derived from the screen rather than extra training rows.
    """
    strategy = config.multifidelity.strategy
    if strategy not in MULTIFIDELITY_STRATEGIES:
        raise ValueError(f"unknown multi-fidelity strategy {strategy!r}")
    allowed = supported_multifidelity(config.model.kind)
    if strategy not in allowed:
        raise ValueError(
            f"multifidelity.strategy={strategy!r} is not available for "
            f"model.kind={config.model.kind!r}; it supports {allowed}"
        )

    weights = SourceWeights(
        screen_pseudo_label=config.multifidelity.screen_weight,
        tdi_arm=config.multifidelity.tdi_weight,
    )
    # The preincubated arm. Omitting it was measured to kill the classification track
    # outright: the curve model's second arm trained on nothing, so
    # predict_is_time_dependent compared an untrained arm against a trained one and
    # returned False for every compound -- MCC 0.0, identical to answering "no" to
    # everything. It is a real measurement of the same compounds under preincubation,
    # so it belongs in the table whatever the model does with it.
    tdi = load("tdi", Path(config.data.directory)) if config.multifidelity.use_tdi_arm else None
    if tdi is not None:
        tdi = tdi[tdi["Molecule_Name"].isin(retained["Molecule_Name"])]
        if tdi.empty:
            # No overlap with the retained compounds. Passing an empty frame through
            # leaves nothing to concatenate; None says "this source is absent", which
            # is what an empty intersection means.
            logger.info("no retained compound has a preincubated measurement")
            tdi = None
    if strategy == "none":
        return build_observations(retained, tdi=tdi, weights=weights), None

    if strategy in SURROGATE_STRATEGIES:
        # Fitted only on retained compounds. Letting it see the holdout would carry
        # those compounds' screen readings into the feature and inflate the score
        # without improving anything on the blinded test set.
        screen = load("screen", Path(config.data.directory))
        surrogate = fit_surrogate(
            screen,
            trainable=set(retained["Molecule_Name"]),
            strategy=strategy,
            n_bits=config.features.fingerprint_bits,
            seed=config.data.seed,
        )
        return build_observations(retained, tdi=tdi, weights=weights), surrogate

    screen = load("screen", Path(config.data.directory))
    kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
    curves = build_observations(retained, tdi=tdi, weights=weights)

    if config.model.kind == "curve":
        # The curve model predicts a response, so the screen enters as the quantity it
        # actually measured -- an inhibited fraction at one concentration. No
        # calibration, and the units match what the loss compares against.
        points = response_observations(kept, config.model.curve.screen_noise)
        points["weight"] = config.multifidelity.screen_weight
        return pd.concat([curves, points[curves.columns]], ignore_index=True), None

    # Every other model predicts a pIC50, so the screen has to be mapped onto that
    # scale first, which is what the isotonic calibration does.
    labels, _ = pseudo_labels(kept, retained)
    return build_observations(retained, screen_labels=labels, tdi=tdi, weights=weights), None


def run(config: ExperimentConfig) -> tuple[pd.DataFrame, PredictiveDistribution]:
    """Fit the configured model and score it on the similarity-matched holdout.

    Returns:
        The score table and the predictive distribution behind it.
    """
    config.validate()
    directory = Path(config.data.directory)
    train = load("inhibition", directory)
    test = load("test", directory)

    mask = similarity_matched_holdout(
        train["SMILES"].tolist(),
        test["SMILES"].tolist(),
        holdout_fraction=config.data.holdout_fraction,
        n_bins=config.data.n_similarity_bins,
        n_rounds=config.data.n_matching_rounds,
        seed=config.data.seed,
    )
    retained, evaluation = train.loc[~mask], train.loc[mask]
    observations, surrogate = assemble_training_data(config, retained)
    logger.info(
        "%s | conditioning %s | multifidelity %s | %d observations",
        config.model.kind,
        config.conditioning.strategy,
        config.multifidelity.strategy,
        len(observations),
    )

    model = build_model(config, surrogate)
    model.fit(observations)
    prediction = model.predict(evaluation["SMILES"].tolist())

    submission = evaluation[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = prediction.mean[:, column]
    scores = score_submission(submission, evaluation)

    tdi = score_tdi(model, evaluation, directory)
    if tdi is not None:
        logger.info(
            "TDI track (ranked by MCC):\n%s", tdi[["MCC_mean", "Accuracy_mean"]].to_string()
        )
    return scores, prediction


def score_tdi(model: object, evaluation: pd.DataFrame, directory: Path) -> pd.DataFrame | None:
    """Score the time-dependent inhibition track, when the model can predict it.

    Only a model of the dose-response curve can. The label is defined by comparing the
    preincubated arm's midpoint against the direct arm's, so predicting it means
    predicting both curves; a model emitting a single pIC50 has nothing to compare.
    Returns None rather than raising for those, since the regression track is still a
    valid run.

    Labels live in their own file and cover a different set of compounds, so the
    holdout is intersected with them rather than assumed to overlap.
    """
    if not hasattr(model, "predict_is_time_dependent"):
        logger.info("%s cannot predict TDI; scoring the regression track only", model.name)
        return None

    labels = load("tdi", directory)
    shared = evaluation.merge(
        labels[["Molecule_Name", *CLASSIFICATION_ENDPOINTS]], on="Molecule_Name"
    )
    if shared.empty:
        logger.info("no holdout compound carries a TDI label")
        return None

    flags = model.predict_is_time_dependent(shared["SMILES"].tolist())
    predicted = pd.DataFrame(
        {
            endpoint: flags[:, list(ISOFORMS).index(endpoint.split("_")[0])]
            for endpoint in CLASSIFICATION_ENDPOINTS
        }
    )
    truth = shared[list(CLASSIFICATION_ENDPOINTS)].astype("boolean")
    logger.info("scoring TDI on %d holdout compounds", len(shared))
    return score_classification(predicted, truth)


def metrics_for_tracking(scores: pd.DataFrame) -> dict[str, float]:
    """Flatten the score table into the scalar metrics a tracker expects."""
    flat = {}
    for endpoint, row in scores.iterrows():
        label = endpoint.split("_")[0]
        for metric, value in row.items():
            if metric.endswith("_mean"):
                flat[f"{label}_{metric.removesuffix('_mean')}"] = float(value)
    return flat
