"""Turning a configuration into a fitted model and a score.

One place where a configuration becomes an experiment, so a sweep and a single run
follow exactly the same path and cannot drift apart.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import REGRESSION_ENDPOINTS
from cypshift.data.assembly import SourceWeights, build_observations
from cypshift.data.challenge import load
from cypshift.data.screen import pseudo_labels
from cypshift.data.splits import similarity_matched_holdout
from cypshift.eval.scoring import score_submission
from cypshift.experiment import ExperimentConfig
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)


def build_model(config: ExperimentConfig) -> object:
    """Construct the model a configuration names.

    Imported lazily so a run needing only boosted trees does not pay for torch, and a
    missing optional dependency fails only when that model is actually requested.
    """
    kind = config.model.kind
    shared = dict(
        conditioning=config.conditioning.strategy,
        n_bits=config.features.fingerprint_bits,
        seed=config.data.seed,
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
    raise ValueError(f"model.kind={kind!r} has no builder")


def assemble_training_data(
    config: ExperimentConfig, retained: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Build the observation table the configured multi-fidelity strategy calls for.

    Returns the observations and, where the strategy produces one, a per-compound
    surrogate feature derived from the screen rather than extra training rows.
    """
    strategy = config.multifidelity.strategy
    # Refuse an unbuilt strategy before touching the data, so a sweep fails in a
    # second rather than after loading and calibrating the screen.
    if strategy in ("predicted_label", "embedding"):
        raise NotImplementedError(
            f"multifidelity.strategy={strategy!r} is declared but not yet built"
        )
    if strategy not in ("none", "extra_rows"):
        raise ValueError(f"unknown multi-fidelity strategy {strategy!r}")

    weights = SourceWeights(
        screen_pseudo_label=config.multifidelity.screen_weight,
        tdi_arm=config.multifidelity.tdi_weight,
    )
    if strategy == "none":
        return build_observations(retained, weights=weights), None

    screen = load("screen", Path(config.data.directory))
    kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
    labels, _ = pseudo_labels(kept, retained)
    return build_observations(retained, screen_labels=labels, weights=weights), None


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
    observations, _ = assemble_training_data(config, retained)
    logger.info(
        "%s | conditioning %s | multifidelity %s | %d observations",
        config.model.kind,
        config.conditioning.strategy,
        config.multifidelity.strategy,
        len(observations),
    )

    model = build_model(config)
    model.fit(observations)
    prediction = model.predict(evaluation["SMILES"].tolist())

    submission = evaluation[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = prediction.mean[:, column]
    return score_submission(submission, evaluation), prediction


def metrics_for_tracking(scores: pd.DataFrame) -> dict[str, float]:
    """Flatten the score table into the scalar metrics a tracker expects."""
    flat = {}
    for endpoint, row in scores.iterrows():
        label = endpoint.split("_")[0]
        for metric, value in row.items():
            if metric.endswith("_mean"):
                flat[f"{label}_{metric.removesuffix('_mean')}"] = float(value)
    return flat
