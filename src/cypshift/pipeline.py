"""Turning a configuration into a fitted model and a score.

One place where a configuration becomes an experiment, so a sweep and a single run
follow exactly the same path and cannot drift apart.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import CLASSIFICATION_ENDPOINTS, ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import OBSERVATION_COLUMNS, SourceWeights, build_observations
from cypshift.data.challenge import load
from cypshift.data.screen import pseudo_labels, response_observations
from cypshift.data.splits import similarity_matched_holdout
from cypshift.data.surrogate import SURROGATE_STRATEGIES, fit_surrogate
from cypshift.eval.calibration import r2_decomposition
from cypshift.eval.classification import score_classification
from cypshift.eval.scoring import score_submission
from cypshift.experiment import (
    CONDITIONING_STRATEGIES,
    SCREEN_STRATEGIES,
    ExperimentConfig,
)
from cypshift.models.base import PredictiveDistribution

logger = logging.getLogger(__name__)


def build_model(
    config: ExperimentConfig,
    surrogate: object | None = None,
    targets: tuple[str, ...] = ISOFORMS,
) -> object:
    """Construct the model a configuration names.

    Imported lazily so a run needing only boosted trees does not pay for torch, and a
    missing optional dependency fails only when that model is actually requested.
    """
    kind = config.model.kind
    if config.conditioning.strategy not in supported_conditioning(kind):
        raise ValueError(
            f"conditioning.strategy={config.conditioning.strategy!r} is not available "
            f"for model.kind={kind!r}; it supports {supported_conditioning(kind)}"
        )
    shared = dict(
        conditioning=config.conditioning.strategy,
        n_bits=config.features.fingerprint_bits,
        fingerprint_radius=config.features.fingerprint_radius,
        physchem=config.features.use_physicochemical,
        mechanism=config.features.use_mechanistic,
        targets=targets,
        seed=config.data.seed,
        surrogate=surrogate,
        training=config.training,
        molecular_embedding=config.features.molecular_embedding,
    )
    if kind == "boosted":
        from cypshift.models.boosted import BoostedTrees

        return BoostedTrees(
            loss=config.model.objective,
            use_structure=config.sources.structure,
            structure_cache=config.features.structure_cache,
            n_members=config.model.n_members,
            n_estimators=config.model.boosted.n_estimators,
            learning_rate=config.model.boosted.learning_rate,
            num_leaves=config.model.boosted.num_leaves,
            min_child_samples=config.model.boosted.min_child_samples,
            subsample=config.model.boosted.subsample,
            colsample_bytree=config.model.boosted.colsample_bytree,
            **shared,
        )
    if kind == "neural":
        from cypshift.models.neural import NeuralNetwork

        return NeuralNetwork(
            loss=config.model.objective,
            embedding_dim=config.conditioning.embedding_dim,
            descriptor_sources=tuple(config.conditioning.descriptor_sources),
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
            targets=targets,
            seed=config.data.seed,
        )
    if kind == "curve":
        from cypshift.models.curve import CurveModel

        return CurveModel(
            targets=targets,
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
            training=config.training,
            molecular_embedding=config.features.molecular_embedding,
        )
    raise ValueError(f"model.kind={kind!r} has no builder")


def supported_conditioning(kind: str) -> tuple[str, ...]:
    """Which input-conditioning strategies a model family can use.

    ``embedding`` looks the (target, assay) pair up in a learned table, which only a
    torch model with a trunk can do -- the neural model. A boosted tree splits on the
    one-hot columns and has nothing to embed; the graph model already feeds its one-hot
    through a linear readout (``x_d``), which is itself a learned projection, so a
    separate embedding would be redundant. The curve model ignores this axis entirely --
    it conditions through its own ``model.curve.conditioning`` (heads/descriptor/film).
    """
    if kind == "neural":
        # The neural model implements every strategy: one-hot (factorised/metadata),
        # a learned table (embedding), the pocket vector (descriptor), per-condition
        # output heads (heads), and descriptor-driven trunk modulation (film).
        return CONDITIONING_STRATEGIES
    # A tree has no trunk to embed into or modulate, and per-enzyme "heads" for a tree
    # is the separate per-target model. The graph model feeds its one-hot through a
    # linear readout already. The curve model conditions on its own axis and ignores
    # this one. So the other families keep the two one-hot input encodings.
    return ("factorised", "metadata")


def supported_screen_strategy(kind: str) -> tuple[str, ...]:
    """Which screen strategies a model family can use when ``sources.screen`` is on.

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
        return ("extra_rows",)
    return SCREEN_STRATEGIES


def assemble_training_data(
    config: ExperimentConfig, retained: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Build the observation table the configured multi-fidelity strategy calls for.

    Returns the observations and, where the strategy produces one, a per-compound
    surrogate feature derived from the screen rather than extra training rows.
    """
    screen_strategy = config.multifidelity.screen_strategy
    if screen_strategy not in SCREEN_STRATEGIES:
        raise ValueError(f"unknown screen strategy {screen_strategy!r}")
    if config.sources.screen:
        allowed = supported_screen_strategy(config.model.kind)
        if screen_strategy not in allowed:
            raise ValueError(
                f"multifidelity.screen_strategy={screen_strategy!r} is not available for "
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
    tdi = load("tdi", Path(config.data.directory)) if config.sources.tdi_arm else None
    if tdi is not None:
        tdi = tdi[tdi["Molecule_Name"].isin(retained["Molecule_Name"])]
        if tdi.empty:
            # No overlap with the retained compounds. Passing an empty frame through
            # leaves nothing to concatenate; None says "this source is absent", which
            # is what an empty intersection means.
            logger.info("no retained compound has a preincubated measurement")
            tdi = None
    surrogate = None
    if not config.sources.screen:
        observations = build_observations(retained, tdi=tdi, weights=weights)
    elif screen_strategy in SURROGATE_STRATEGIES:
        # Fitted only on retained compounds. Letting it see the holdout would carry
        # those compounds' screen readings into the feature and inflate the score
        # without improving anything on the blinded test set.
        screen = load("screen", Path(config.data.directory))
        surrogate = fit_surrogate(
            screen,
            trainable=set(retained["Molecule_Name"]),
            strategy=screen_strategy,
            n_bits=config.features.fingerprint_bits,
            hidden=config.multifidelity.surrogate_hidden,
            seed=config.data.seed,
        )
        observations = build_observations(retained, tdi=tdi, weights=weights)
    else:
        screen = load("screen", Path(config.data.directory))
        kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
        if config.model.kind == "curve":
            # The curve model predicts a response, so the screen enters as the quantity
            # it actually measured -- an inhibited fraction at one concentration. No
            # calibration, and the units match what the loss compares against.
            curves = build_observations(retained, tdi=tdi, weights=weights)
            points = response_observations(kept, config.model.curve.screen_noise)
            points["weight"] = config.multifidelity.screen_weight
            observations = pd.concat([curves, points[curves.columns]], ignore_index=True)
        else:
            # Every other model predicts a pIC50, so the screen has to be mapped onto
            # that scale first, which is what the isotonic calibration does.
            labels, _ = pseudo_labels(kept, retained)
            observations = build_observations(
                retained, screen_labels=labels, tdi=tdi, weights=weights
            )

    auxiliary = auxiliary_observations(config, retained)
    if not auxiliary.empty:
        observations = pd.concat([observations, auxiliary[observations.columns]], ignore_index=True)
    return observations, surrogate


def auxiliary_observations(config: ExperimentConfig, retained: pd.DataFrame) -> pd.DataFrame:
    """Weighted training rows from auxiliary enzymes, per the auxiliary config.

    Selects enzymes whose binding site is within ``auxiliary.max_distance`` of a scored
    one, pulls their cleaned ChEMBL potencies, and weights each enzyme's rows by
    distance (:mod:`cypshift.data.auxiliary`). Rows carry the auxiliary enzyme as their
    ``target``; the model's encoding vocabulary is widened to fit them while predictions
    stay on the four scored endpoints.

    Returns an empty frame -- and the run trains on the challenge data alone -- when the
    config admits nothing or the data is unreachable (no network and no cache), so a run
    never fails for lack of an auxiliary source.
    """
    from cypshift.config import ISOFORMS
    from cypshift.data import chembl, novartis, octant, sair, tox21, veith
    from cypshift.data.auxiliary import select_auxiliaries, site_distances
    from cypshift.data.sequences import ACCESSIONS, sequence_for

    all_loaders = {
        "chembl": chembl.load_auxiliary,
        "sair": sair.load_auxiliary,
        "octant": octant.load_auxiliary,
        "veith": veith.load_auxiliary,
        "tox21": tox21.load_auxiliary,
        "novartis": novartis.load_auxiliary,
    }
    # Each source's detection chemistry decides its assay context, so a luminescent AC50 is
    # never pooled onto a fluorogenic IC50. Sources sharing a chemistry share a context and
    # are pooled within it (veith+tox21, chembl+sair).
    assay_by_source = {
        "chembl": "literature_ic50",
        "sair": "literature_ic50",
        "octant": "fluorogenic_pic50",
        "veith": "luminescent_ac50",
        "tox21": "luminescent_ac50",
        "novartis": "reversible_pic50",
    }
    datasets = [name for name in all_loaders if getattr(config.sources, name)]

    scored = tuple(name.split("_")[0] for name in ISOFORMS)
    empty = pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    if not datasets:
        return empty

    # Deduplicate against the WHOLE challenge -- every train and test compound, not just
    # this split's retained rows -- so a source row matching a held-out or blinded-test
    # compound can never leak into training. Measured: 246 challenge compounds sit in
    # SAIR, so this is not hypothetical.
    directory = Path(config.data.directory)
    challenge = pd.concat(
        [load("inhibition", directory)["SMILES"], load("test", directory)["SMILES"]]
    )

    # (enzyme, weight) pairs to pull, on one distance axis. The scored enzymes sit at
    # distance 0 -- their own rows from the outside source, weighted by scored_weight --
    # and non-scored relatives are admitted out to auxiliary.max_distance and weighted by
    # how far their binding site is. max_distance = 0 pulls the scored enzymes alone.
    to_pull: list[tuple[str, float]] = [(name, config.auxiliary.scored_weight) for name in scored]
    try:
        candidates = {g: sequence_for(g) for g in ACCESSIONS if g not in scored}
        admitted = select_auxiliaries(
            site_distances(scored, candidates),
            max_distance=config.auxiliary.max_distance,
            weighting=config.auxiliary.weighting,
            scale=config.auxiliary.scale,
        )
        to_pull.extend((aux.name, aux.weight * config.auxiliary.target_weight) for aux in admitted)
    except Exception as error:  # noqa: BLE001 -- any failure means "no relatives"
        logger.info("auxiliary relative selection unavailable (%s)", error)
    if not to_pull:
        return empty

    frames = []
    for enzyme, weight in to_pull:
        # A scored enzyme's own outside rows sit one tier from the endpoint; a relative's
        # further out. The curriculum trains distant tiers before near ones.
        tier = "near_aux" if enzyme in scored else "distant_aux"
        # Group the enabled datasets by their assay context, so sources of one detection
        # chemistry (veith+tox21, chembl+sair) pool together while different chemistries
        # stay distinct rows the calibration can offset apart.
        # Only the qHTS loaders take the inactive-inclusion flag; the rest ignore it.
        extra_kwargs = {"veith": {"include_inactives": config.auxiliary.include_qhts_inactives}}
        by_assay: dict[str, list[pd.DataFrame]] = {}
        for name in datasets:
            try:
                loaded = all_loaders[name](enzyme, challenge, **extra_kwargs.get(name, {}))
            except Exception as error:  # noqa: BLE001
                logger.info("no %s data for %s (%s); skipping", name, enzyme, error)
                continue
            if loaded.empty:
                continue
            # Give every source the same columns, so pooling and row-building are uniform:
            # a source without intervals collapses its bounds to the point and is uncensored.
            for column, default in (("conf_low", loaded["value"]), ("conf_high", loaded["value"])):
                if column not in loaded.columns:
                    loaded[column] = default
            if "is_censored" not in loaded.columns:
                loaded["is_censored"] = False
            by_assay.setdefault(assay_by_source[name], []).append(loaded)

        for assay, gathered in by_assay.items():
            data = pd.concat(gathered, ignore_index=True)
            if len(gathered) > 1:
                # Two sources of the same chemistry: pool shared compounds by median,
                # which drops per-source intervals, so bounds collapse to the point. A
                # compound stays censored only if every pooled source censored it.
                data = data.groupby("SMILES", as_index=False).agg(
                    value=("value", "median"), is_censored=("is_censored", "all")
                )
            value = data["value"].to_numpy(float)
            # Keep a real credible interval where the source ships one (Octant does),
            # otherwise the bounds collapse to the point -- no invented tolerance.
            low = data["conf_low"].to_numpy(float) if "conf_low" in data.columns else value
            high = data["conf_high"].to_numpy(float) if "conf_high" in data.columns else value
            censored = (
                data["is_censored"].to_numpy(bool)
                if "is_censored" in data.columns
                else np.zeros(len(data), dtype=bool)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "Molecule_Name": [f"{enzyme}:{assay}:{i}" for i in range(len(data))],
                        "SMILES": data["SMILES"].to_numpy(),
                        "target": enzyme,
                        "assay": assay,
                        "value": value,
                        "conf_low": low,
                        "conf_high": high,
                        "weight": weight,
                        "is_censored": censored,
                        "tier": tier,
                    }
                )
            )
    if not frames:
        return empty
    out = pd.concat(frames, ignore_index=True)
    logger.info("added %d auxiliary rows from %d enzymes", len(out), out["target"].nunique())
    return out


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
    scored = list(ISOFORMS)
    extra = [t for t in dict.fromkeys(observations["target"]) if t not in scored]
    targets = tuple(scored + sorted(extra))
    logger.info(
        "%s | conditioning %s | multifidelity %s | %d observations",
        config.model.kind,
        config.conditioning.strategy,
        config.multifidelity.screen_strategy,
        len(observations),
    )

    model = build_model(config, surrogate, targets=targets)
    model.fit(observations)
    prediction = model.predict(evaluation["SMILES"].tolist())

    submission = evaluation[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = prediction.mean[:, column]
    scores = score_submission(submission, evaluation)
    _log_r2_decomposition(submission, evaluation)

    tdi = score_tdi(model, evaluation, directory)
    if tdi is not None:
        logger.info(
            "TDI track (ranked by MCC):\n%s", tdi[["MCC_mean", "Accuracy_mean"]].to_string()
        )
    return scores, prediction


def _log_r2_decomposition(submission: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Log R² split into its ranking ceiling and the spread and offset it loses.

    Ranking (rho) transfers from this holdout to the blinded test; the spread and offset
    terms do not, since the holdout's potency distribution differs from the test's. The
    split says which part of a change is a real modelling gain rather than a calibration
    the holdout cannot vouch for.
    """
    rows = {}
    for endpoint in REGRESSION_ENDPOINTS:
        measured = evaluation[endpoint].notna().to_numpy()
        rows[endpoint.split("_")[0]] = r2_decomposition(
            evaluation.loc[measured, endpoint].to_numpy(float),
            submission.loc[measured, endpoint].to_numpy(float),
        )
    table = pd.DataFrame(rows).T[
        ["rho", "spread_ratio", "offset", "r2", "ranking_ceiling", "spread_loss", "offset_loss"]
    ]
    logger.info("R² decomposition (rho transfers; spread/offset do not):\n%s", table.round(3))


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
