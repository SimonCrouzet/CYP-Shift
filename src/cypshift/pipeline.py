"""Turning a configuration into a fitted model and a score.

One place where a configuration becomes an experiment, so a sweep and a single run
follow exactly the same path and cannot drift apart.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import CLASSIFICATION_ENDPOINTS, ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import (
    OBSERVATION_COLUMNS,
    RESPONSE_ASSAY,
    SCREEN_LOG2FC_ASSAY,
    SCREEN_RESPONSE_ASSAY,
    SourceWeights,
    build_observations,
)
from cypshift.data.challenge import load
from cypshift.data.screen import pseudo_labels, response_observations
from cypshift.data.screen_model import (
    SCREEN_MODEL_STRATEGIES,
    fit_response_model,
    fit_screen_model,
    stack_helper_models,
)
from cypshift.data.sources import supplying
from cypshift.data.splits import similarity_matched_holdout, stratified_scaffold_folds
from cypshift.eval.calibration import r2_decomposition
from cypshift.eval.classification import score_classification
from cypshift.eval.scoring import score_submission
from cypshift.experiment import (
    CONDITIONING_STRATEGIES,
    RESPONSE_FEATURE_KINDS,
    SCREEN_STRATEGIES,
    ExperimentConfig,
)
from cypshift.models.base import PredictiveDistribution, training_stages
from cypshift.models.monitor import reference_error, write_curves

logger = logging.getLogger(__name__)


def build_model(
    config: ExperimentConfig,
    screen_model: object | None = None,
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
        use_fingerprint=config.features.use_fingerprint,
        physchem=config.features.use_physicochemical,
        mechanism=config.features.use_mechanistic,
        targets=targets,
        seed=config.data.seed,
        screen_model=screen_model,
        training=config.training,
        molecular_embedding=config.features.molecular_embedding,
        ionization_geometry=config.features.use_ionization_geometry,
        ionization_cache=config.features.ionization_cache,
    )
    if kind == "boosted":
        from cypshift.models.boosted import BoostedTrees

        return BoostedTrees(
            loss=config.model.objective,
            tau=config.model.objective_tau,
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
            tau=config.model.objective_tau,
            embedding_dim=config.conditioning.embedding_dim,
            descriptor_sources=tuple(config.conditioning.descriptor_sources),
            hidden=config.model.neural.hidden,
            dropout=config.model.neural.dropout,
            epochs=config.model.neural.epochs,
            batch_size=config.model.neural.batch_size,
            learning_rate=config.model.neural.learning_rate,
            n_forward_passes=config.model.neural.forward_passes,
            tdi_shift_threshold=config.model.neural.tdi_shift_threshold,
            tdi_assay_floor=config.model.neural.tdi_assay_floor,
            **shared,
        )
    if kind == "graph":
        from cypshift.models.graph import GraphNetwork, SeparatedTrunkGraph

        graph_cls = SeparatedTrunkGraph if config.model.graph.separate_trunks else GraphNetwork
        # Which measurements the graph's response heads train on: a qHTS max-response
        # (sources.qhts_representation) and/or the single-concentration screen fraction
        # (multifidelity.screen_strategy='response_head'). They coexist -- each gets its own
        # response column -- so the panel can carry both at once; an empty tuple leaves the
        # response heads off entirely.
        response_assays = []
        if config.sources.qhts_representation in ("max_response", "both"):
            response_assays.append(RESPONSE_ASSAY)
        if config.sources.screen:
            if config.multifidelity.screen_strategy == "response_head":
                response_assays.append(SCREEN_RESPONSE_ASSAY)
            elif config.multifidelity.screen_strategy == "log2fc_response":
                response_assays.append(SCREEN_LOG2FC_ASSAY)
        return graph_cls(
            loss=config.model.objective,
            tau=config.model.objective_tau,
            point_loss_weight=config.model.point_loss_weight,
            ranking_loss_weight=config.model.ranking_loss_weight,
            hidden=config.model.graph.hidden,
            depth=config.model.graph.depth,
            dropout=config.model.graph.dropout,
            epochs=config.model.graph.epochs,
            batch_size=config.model.graph.batch_size,
            num_workers=config.model.graph.num_workers,
            n_members=config.model.n_members,
            ensemble=config.model.graph.ensemble,
            pretraining=config.model.graph.pretraining,
            pretraining_cache=config.model.graph.pretraining_cache,
            pretraining_finetune=config.model.graph.pretraining_finetune,
            source_heads=config.model.graph.source_heads,
            adapt_target=config.model.graph.adapt_target,
            adapt_hidden=config.model.graph.adapt_hidden,
            per_target_heads=config.model.graph.per_target_heads,
            per_measurement_heads=config.model.graph.per_measurement_heads,
            ffn_hidden=config.model.graph.ffn_hidden,
            ffn_layers=config.model.graph.ffn_layers,
            batch_norm=config.model.graph.batch_norm,
            aggregation=config.model.graph.aggregation,
            learning_rate=config.model.graph.learning_rate,
            warmup_epochs=config.model.graph.warmup_epochs,
            matmul_precision=config.model.graph.matmul_precision,
            early_stopping_patience=config.model.graph.early_stopping_patience,
            stages=tuple(
                training_stages(
                    config.training,
                    config.model.graph.epochs,
                    config.model.graph.learning_rate,
                )
            ),
            response_assays=tuple(response_assays),
            tdi_shift_threshold=config.model.graph.tdi_shift_threshold,
            tdi_assay_floor=config.model.graph.tdi_assay_floor,
            physchem=config.features.use_physicochemical,
            mechanism=config.features.use_mechanistic,
            ionization_geometry=config.features.use_ionization_geometry,
            ionization_cache=config.features.ionization_cache,
            use_structure=config.sources.structure,
            structure_cache=config.features.structure_cache,
            conditioning=config.conditioning.strategy,
            targets=targets,
            seed=config.data.seed,
        )
    if kind == "curve":
        if config.model.curve.backbone == "graph":
            # The cross-model track: the dose-response head on a chemprop D-MPNN backbone
            # instead of the fingerprint MLP. The encoder learns the representation, so the
            # fingerprint feature flags do not apply; physchem/mechanism/ionization ride along
            # as x_d. The screen still enters as observation rows (same assembly path), so no
            # screen_model input column is needed.
            from cypshift.models.curve import GraphCurveModel

            return GraphCurveModel(
                targets=targets,
                hidden=config.model.curve.hidden,
                depth=config.model.curve.graph_depth,
                dropout=config.model.curve.dropout,
                pretraining=config.model.curve.pretraining,
                pretraining_cache=config.model.curve.pretraining_cache,
                pretraining_finetune=config.model.curve.pretraining_finetune,
                physchem=config.features.use_physicochemical,
                mechanism=config.features.use_mechanistic,
                ionization_geometry=config.features.use_ionization_geometry,
                ionization_cache=config.features.ionization_cache,
                epochs=config.model.curve.epochs,
                batch_size=config.model.curve.batch_size,
                learning_rate=config.model.curve.learning_rate,
                num_workers=config.model.graph.num_workers,
                n_forward_passes=config.model.curve.forward_passes,
                conditioning=config.model.curve.conditioning,
                descriptor_sources=tuple(config.model.curve.descriptor_sources),
                tdi_shift_threshold=config.model.curve.tdi_shift_threshold,
                tdi_assay_floor=config.model.curve.tdi_assay_floor,
                loss_tau=config.model.curve.loss_tau,
                seed=config.data.seed,
            )
        from cypshift.models.curve import CurveModel

        return CurveModel(
            targets=targets,
            n_bits=config.features.fingerprint_bits,
            fingerprint_radius=config.features.fingerprint_radius,
            use_fingerprint=config.features.use_fingerprint,
            physchem=config.features.use_physicochemical,
            mechanism=config.features.use_mechanistic,
            hidden=config.model.curve.hidden,
            dropout=config.model.curve.dropout,
            trunk_depth=config.model.curve.trunk_depth,
            epochs=config.model.curve.epochs,
            batch_size=config.model.curve.batch_size,
            learning_rate=config.model.curve.learning_rate,
            n_forward_passes=config.model.curve.forward_passes,
            conditioning=config.model.curve.conditioning,
            descriptor_sources=tuple(config.model.curve.descriptor_sources),
            tdi_shift_threshold=config.model.curve.tdi_shift_threshold,
            tdi_assay_floor=config.model.curve.tdi_assay_floor,
            loss_tau=config.model.curve.loss_tau,
            seed=config.data.seed,
            screen_model=screen_model,
            training=config.training,
            molecular_embedding=config.features.molecular_embedding,
            ionization_geometry=config.features.use_ionization_geometry,
            ionization_cache=config.features.ionization_cache,
        )
    if kind == "tfm":
        from cypshift.models.tfm import TabularFoundationModel

        # Only the scored isoforms: the TFM fits one regressor per isoform and does not use
        # auxiliary-enzyme conditioning, so extra targets would be fit but never read.
        return TabularFoundationModel(
            targets=ISOFORMS,
            backend=config.model.tfm.backend,
            n_bits=config.features.fingerprint_bits,
            fingerprint_radius=config.features.fingerprint_radius,
            use_fingerprint=config.features.use_fingerprint,
            physchem=config.features.use_physicochemical,
            mechanism=config.features.use_mechanistic,
            molecular_embedding=config.features.molecular_embedding,
            ionization_geometry=config.features.use_ionization_geometry,
            ionization_cache=config.features.ionization_cache,
            screen_model=screen_model,
            pca_components=config.model.tfm.pca_components,
            seed=config.data.seed,
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

    The screen_model strategies fit a model to the screen and hand its output, or its
    hidden layer, to the real model as an input column. That is a way of using a cheap
    measurement when the model cannot consume the measurement itself, and two families
    do not need it:

    ``curve``
        The screen is a reading of inhibition at 4.95e-5 M, which is a point on the very
        curve this model predicts. It enters through ``extra_rows`` as an observation in
        its own units, with no calibration and nothing fitted in between. Passing the
        same readings through a screen_model instead would replace a direct measurement
        with a prediction of it, which is strictly worse and was the reason for
        modelling a curve in the first place.
    ``graph``
        Consumes molecular graphs rather than a dense feature matrix, so a screen_model
        column has nowhere to go.

    Refusing beats accepting the setting and quietly training without it, which would
    report a comparison that never happened.

    The graph is the exception to its own "no screen_model" rule: it cannot take a
    screen_model feature column, but it *can* take the screen as a response task through
    ``response_head``, so that strategy is offered to it alongside ``extra_rows``.
    """
    if kind == "curve":
        return ("extra_rows",)
    if kind == "graph":
        return ("extra_rows", "response_head", "log2fc_response")
    return SCREEN_STRATEGIES


def assemble_training_data(
    config: ExperimentConfig,
    retained: pd.DataFrame,
    exclude_names: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray] | None]:
    """Build the observation table the configured multi-fidelity strategy calls for.

    ``exclude_names`` are compounds kept out of the preincubated (TDI) assay_condition -- the
    holdout
    and blinded-test compounds, so their TDI measurement cannot leak. Excluding by name
    rather than intersecting with the direct-labelled set keeps the TDI-only cohort (around
    1,238 CYP3A4 negatives have a preincubated assay_condition but no direct curve) in training.

    Returns the observations and, where the strategy produces one, a per-compound
    screen_model feature derived from the screen rather than extra training rows.
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
        tdi_assay_condition=config.multifidelity.tdi_weight,
    )
    # The preincubated assay_condition. Omitting it was measured to kill the classification track
    # outright: the curve model's second assay_condition trained on nothing, so
    # predict_is_time_dependent compared an untrained assay_condition against a trained one and
    # returned False for every compound -- MCC 0.0, identical to answering "no" to
    # everything. It is a real measurement of the same compounds under preincubation,
    # so it belongs in the table whatever the model does with it.
    tdi = load("tdi", Path(config.data.directory)) if config.sources.tdi_assay_condition else None
    if tdi is not None:
        # Keep every TDI-cohort compound except the held-out and test ones; intersecting
        # with the direct-labelled set instead would drop the ~1,238 compounds that carry a
        # preincubated assay_condition but were never promoted to a direct curve.
        if exclude_names:
            tdi = tdi[~tdi["Molecule_Name"].isin(exclude_names)]
        if tdi.empty:
            logger.info("no preincubated measurement survives the exclusions")
            tdi = None
    screen_model = None
    if not config.sources.screen:
        observations = build_observations(retained, tdi=tdi, weights=weights)
    elif screen_strategy in SCREEN_MODEL_STRATEGIES:
        # Fitted only on retained compounds. Letting it see the holdout would carry
        # those compounds' screen readings into the feature and inflate the score
        # without improving anything on the blinded test set.
        screen = load("screen", Path(config.data.directory))
        screen_model = fit_screen_model(
            screen,
            trainable=set(retained["Molecule_Name"]),
            strategy=screen_strategy,
            n_bits=config.features.fingerprint_bits,
            hidden=config.multifidelity.screen_model_hidden,
            seed=config.data.seed,
        )
        observations = build_observations(retained, tdi=tdi, weights=weights)
    else:
        screen = load("screen", Path(config.data.directory))
        kept = screen[screen["Molecule_Name"].isin(retained["Molecule_Name"])]
        if screen_strategy in ("response_head", "log2fc_response"):
            # Keep the screen as the response it measured and route it to a graph response head
            # (validated graph-only) under its own assay tag, so it informs the model without
            # being compressed onto a pIC50 by the isotonic map first. ``response_head`` sends
            # the inhibited fraction; ``log2fc_response`` sends the raw measured log2 fold-change.
            if screen_strategy == "response_head":
                assay, kind = SCREEN_RESPONSE_ASSAY, "fraction"
            else:
                assay, kind = SCREEN_LOG2FC_ASSAY, "log2fc"
            base = build_observations(retained, tdi=tdi, weights=weights)
            points = response_observations(kept, assay=assay, kind=kind)
            points["weight"] = config.multifidelity.screen_weight
            if "source" not in points.columns:
                points["source"] = "screen"
            observations = pd.concat([base, points[base.columns]], ignore_index=True)
        elif config.model.kind == "curve":
            # The curve model predicts a response, so the screen enters as the quantity
            # it actually measured -- an inhibited fraction at one concentration. No
            # calibration, and the units match what the loss compares against.
            curves = build_observations(retained, tdi=tdi, weights=weights)
            points = response_observations(kept, config.model.curve.screen_noise)
            points["weight"] = config.multifidelity.screen_weight
            if "source" not in points.columns:
                points["source"] = "screen"
            observations = pd.concat([curves, points[curves.columns]], ignore_index=True)
        else:
            # Every other model predicts a pIC50, so the screen has to be mapped onto
            # that scale first, which is what the isotonic calibration does.
            labels, _ = pseudo_labels(kept, retained)
            observations = build_observations(
                retained, screen_labels=labels, tdi=tdi, weights=weights
            )

    if response_feature_route(config):
        auxiliary, response = auxiliary_frames(config, retained)
        if response.empty:
            logger.warning(
                "sources.qhts_representation=%r was requested but no max-response rows were "
                "realised; the response helper model is skipped",
                config.sources.qhts_representation,
            )
        else:
            response_model = fit_response_model(
                response,
                n_bits=config.features.fingerprint_bits,
                hidden=config.multifidelity.screen_model_hidden,
                seed=config.data.seed,
            )
            screen_model = stack_helper_models(screen_model, response_model)
    else:
        auxiliary = auxiliary_observations(config, retained)
    if not auxiliary.empty:
        # Default the source for any frame that predates the column (external or test frames);
        # it is only read under source-stratified conditioning, where real sources are present.
        if "source" not in auxiliary.columns:
            auxiliary = auxiliary.assign(source="_pooled")
        observations = pd.concat([observations, auxiliary[observations.columns]], ignore_index=True)
    if config.training.inverse_variance_weighting:
        observations = weight_by_inverse_variance(
            observations, response_assays=precision_response_assays(config)
        )
    if config.training.macro_balance:
        observations = balance_macro_weights(observations)
    log_realized_inventory(observations)
    return observations, screen_model


def log_realized_inventory(observations: pd.DataFrame) -> pd.DataFrame:
    """Log what the training table actually holds -- rows per target for each (source, assay), and
    the total weight each carries -- and return it.

    A requested source can contribute fewer endpoint blocks than its name suggests (an auxiliary
    loader that fails is logged and skipped, and a response cache can lack an enzyme), so a
    configuration name does not show which data a run trained on; this line does.
    """
    source = (
        observations["source"].fillna("challenge")
        if "source" in observations.columns
        else pd.Series("challenge", index=observations.index)
    )
    keys = [source.rename("source"), observations["assay"], observations["target"]]
    counts = observations.groupby(keys).size().unstack(fill_value=0)
    counts["weight"] = observations.groupby(keys[:2])["weight"].sum().round(1)
    logger.info(
        "realized training table: %d rows (rows per target; total weight):\n%s",
        len(observations),
        counts.to_string(),
    )
    return counts


#: Assays whose label is a response quantity (an inhibited fraction, a raw log2 fold-change, or a
#: qHTS maximum response), not a potency with a credible interval. Their ``conf_low``/``conf_high``
#: are a construction band or a degenerate ``low == high == value``, in units that are not the
#: pIC50 interval, so inverse-variance precision weighting must not read them as a tight potency.
_RESPONSE_ASSAYS = frozenset({RESPONSE_ASSAY, SCREEN_RESPONSE_ASSAY, SCREEN_LOG2FC_ASSAY})


#: The screen tag the curve model's physical screen rows carry. The tag is shared with the
#: legacy isotonic pseudo-potency rows of the other families, so it is a response only for the
#: curve model; see :func:`precision_response_assays`.
_SCREEN_ASSAY = "direct_inhibition_screen"


def precision_response_assays(config: ExperimentConfig) -> frozenset[str]:
    """The assays whose rows are responses, not potencies, for this configuration.

    ``direct_inhibition_screen`` is backend-dependent. The curve model reads the screen as the
    inhibited fraction at one concentration (a response in [0, 1] with a fixed +-``screen_noise``
    construction band), whereas every other family maps it through the isotonic calibration onto
    a pIC50 and keeps it as a potency row, which is legitimately precision-weighted. Excluding
    the tag globally would change the latter, so only the curve model adds it.
    """
    if config.model.kind == "curve":
        return _RESPONSE_ASSAYS | {_SCREEN_ASSAY}
    return _RESPONSE_ASSAYS


def weight_by_inverse_variance(
    observations: pd.DataFrame,
    floor: float = 0.1,
    response_assays: frozenset[str] = _RESPONSE_ASSAYS,
) -> pd.DataFrame:
    """Scale each measured potency interval's weight by the inverse of its interval variance.

    A measurement's credible-interval width is a precision proxy: a tight band is a confident
    label and should pull harder than a wide one. The factor is ``1 / max(width/2, floor)**2``
    (``floor`` caps the influence of an implausibly tight band), then renormalised so the total
    weight of the affected rows is unchanged -- this redistributes weight by precision, it does
    not inflate it.

    Only genuinely measured, two-sided potency intervals are eligible, which the earlier version
    did not enforce. Three kinds of row are now excluded explicitly:

    * **Censored (one-sided) rows**, which have no two-sided interval.
    * **Response-task rows** (``response_assays``, by default ``_RESPONSE_ASSAYS``; the curve
      model's screen fraction is added by :func:`precision_response_assays`): the screen
      fraction carries a fixed +-0.1 construction band and the qHTS response carries
      ``low == high == value``, both in response
      units, not pIC50; reading those as tight potencies gave every one the maximum precision
      factor and swamped the scored potency rows.
    * **Zero-width rows** (``width <= 0``): a missing credible interval is filled with
      ``low = high = value`` upstream, which means uncertainty was *not supplied*, not that the
      measurement is near-exact -- so it must not collect the ``1/floor**2`` maximum factor.
    """
    out = observations.copy()
    low = out["conf_low"].to_numpy(float)
    high = out["conf_high"].to_numpy(float)
    width = high - low
    censored = out["is_censored"].to_numpy(bool) if "is_censored" in out.columns else False
    is_response = (
        out["assay"].isin(response_assays).to_numpy()
        if "assay" in out.columns
        else np.zeros(len(out), bool)
    )
    eligible = np.isfinite(width) & (width > 0) & ~np.asarray(censored) & ~is_response
    if not eligible.any():
        return out
    factor = 1.0 / np.maximum(width[eligible] / 2.0, floor) ** 2
    weight = out.loc[eligible, "weight"].to_numpy(float)
    rescaled = weight * factor
    total = rescaled.sum()
    if total > 0:
        rescaled *= weight.sum() / total  # preserve the affected rows' total weight mass
    out.loc[eligible, "weight"] = rescaled
    return out


def balance_macro_weights(observations: pd.DataFrame) -> pd.DataFrame:
    """Rescale each scored endpoint's rows so the four carry equal total weight.

    The leaderboard macro-averages the four endpoints, but a pooled fit is dominated by
    the data-rich ones (CYP3A4 has nearly twice CYP2C9's labels). Scaling each scored
    target's weights to a common total aligns training with the metric. Non-scored
    (relative) rows are left untouched, and the fix only bites now that the boosted
    objective actually applies the weights.
    """
    scored = observations["target"].isin(ISOFORMS)
    if not scored.any():
        return observations
    totals = observations.loc[scored].groupby("target")["weight"].sum()
    scale = (totals.mean() / totals).to_dict()
    observations = observations.copy()
    factors = observations.loc[scored, "target"].map(scale).to_numpy()
    observations.loc[scored, "weight"] = observations.loc[scored, "weight"].to_numpy() * factors
    return observations


def cap_per_endpoint_budget(
    auxiliary: pd.DataFrame, retained: pd.DataFrame, budget: float
) -> pd.DataFrame:
    """Rescale each scored enzyme's near-auxiliary rows to ``budget`` times its label weight.

    For each scored enzyme, multiplies its ``near_aux`` rows' weights by a common factor so
    their sum equals ``budget`` times the enzyme's challenge-label count (challenge rows carry
    dose_response weight 1.0). Relatives (``distant_aux``) are left on their distance weighting;
    an enzyme with no near-auxiliary rows is left untouched.
    """
    out = auxiliary.copy()
    for enzyme in ISOFORMS:
        rows = (out["target"] == enzyme) & (out["data_space"] == "near_aux")
        current = float(out.loc[rows, "weight"].sum())
        if current <= 0:
            continue
        n_labels = int(retained[f"{enzyme}_pIC50_direct_inhibition"].notna().sum())
        target_mass = budget * n_labels  # the challenge rows carry dose_response weight 1.0
        scale = target_mass / current
        out.loc[rows, "weight"] = out.loc[rows, "weight"].to_numpy() * scale
        logger.info(
            "per-endpoint budget: %s near-aux mass %.1f -> %.1f (%d challenge labels x %.2f)",
            enzyme,
            current,
            target_mass,
            n_labels,
            budget,
        )
    return out


def pool_same_assay(data: pd.DataFrame) -> pd.DataFrame:
    """Pool one assay's rows across sources by canonical structure, preserving censoring.

    A real (uncensored) measurement supersedes a censored detection-limit bound rather than
    being median-pooled with it: a group with any uncensored row becomes the median of the
    uncensored values, a wholly censored group keeps its tightest bound (the lowest value).
    A censored bound and an uncensored point are never averaged into a value neither source
    measured. Grouping is on canonical SMILES, so ``CCO`` and ``OCC`` are one compound.
    """
    from cypshift.data.chembl import canonical

    work = data.copy()
    work["_canonical"] = canonical(work["SMILES"]).to_numpy()
    work = work.dropna(subset=["_canonical"])
    rows = []
    for _, group in work.groupby("_canonical"):
        uncensored = group[~group["is_censored"].astype(bool)]
        if not uncensored.empty:
            value, censored = float(uncensored["value"].median()), False
        else:
            value, censored = float(group["value"].min()), True
        rows.append(
            {
                "SMILES": group["SMILES"].iloc[0],
                "value": value,
                "conf_low": value,
                "conf_high": value,
                "is_censored": censored,
            }
        )
    return pd.DataFrame(rows)


def response_feature_route(config: ExperimentConfig) -> bool:
    """Whether the qHTS max-response reaches the model as helper-model features.

    True for the model kinds without a response head when ``sources.qhts_representation`` asks
    for the max-response; the graph consumes it through its own response head instead.
    """
    return (
        any(getattr(config.sources, name) for name in supplying("max_response"))
        and config.sources.qhts_representation in ("max_response", "both")
        and config.model.kind in RESPONSE_FEATURE_KINDS
    )


def auxiliary_observations(config: ExperimentConfig, retained: pd.DataFrame) -> pd.DataFrame:
    """The weighted auxiliary training rows (:func:`auxiliary_frames`, first frame)."""
    return auxiliary_frames(config, retained)[0]


def auxiliary_frames(
    config: ExperimentConfig, retained: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted training rows from auxiliary enzymes, per the auxiliary config, and the qHTS
    max-response rows held apart for a helper model.

    Selects enzymes whose binding site is within ``auxiliary.max_distance`` of a scored
    one, pulls their cleaned ChEMBL potencies, and weights each enzyme's rows by
    distance (:mod:`cypshift.data.auxiliary`). Rows carry the auxiliary enzyme as their
    ``target``; the model's encoding vocabulary is widened to fit them while predictions
    stay on the four scored endpoints.

    Returns an empty first frame -- and the run trains on the challenge data alone -- when the
    config admits nothing or the data is unreachable (no network and no cache), so a run
    never fails for lack of an auxiliary source. The second frame is empty unless the model
    reads the qHTS max-response through a helper model (:func:`response_feature_route`); then it
    holds the response rows and the first frame holds none of them.
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
    # A source whose loader supplies a max-response can enter as a 0-1 response on its own assay
    # instead of a pooled pIC50 (``max_response``); the graph reads it through its response head,
    # the other kinds through a helper model. ``both`` keeps the pooled pIC50 assay *and*
    # additionally emits the response rows, so the one panel carries potency and response at once.
    response_mode = config.sources.qhts_representation == "max_response"
    both_mode = config.sources.qhts_representation == "both"
    # Source-stratified conditioning keeps same-chemistry sources (veith+tox21, chembl+sair)
    # as separate rows instead of pooling them, so the graph can condition on the source.
    source_heads = config.model.kind == "graph" and config.model.graph.source_heads
    datasets = [name for name in all_loaders if getattr(config.sources, name)]
    # The sources whose loader supplies a max-response; the others keep their pIC50 rows.
    with_response = [name for name in datasets if name in supplying("max_response")]
    if response_mode:
        for name in with_response:
            assay_by_source[name] = RESPONSE_ASSAY

    scored = tuple(name.split("_")[0] for name in ISOFORMS)
    empty = pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    if not datasets:
        return empty, empty

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
        return empty, empty

    frames = []
    # Track each requested source's outcome so a named source-ablation that realised no
    # rows is visible, and a parser failure is not mistaken for "no data for this enzyme".
    contributed = {name: 0 for name in datasets}
    raised: dict[str, str] = {}
    for enzyme, weight in to_pull:
        # A scored enzyme's own outside rows sit one data_space from the endpoint; a relative's
        # further out. The curriculum trains distant data_spaces before near ones.
        data_space = "near_aux" if enzyme in scored else "distant_aux"
        # Group the enabled datasets by their assay context, so sources of one detection
        # chemistry (veith+tox21, chembl+sair) pool together while different chemistries
        # stay distinct rows the calibration can offset apart.
        # The qHTS loaders take the representation and inactive flags; the others ignore them.
        potency_kwargs = {
            "veith": {"include_inactives": config.auxiliary.include_qhts_inactives},
            "tox21": {"include_inactives": config.auxiliary.tox21_include_inactives},
        }
        extra_kwargs = {
            name: {"representation": "max_response"}
            if response_mode and name in with_response
            else kwargs
            for name, kwargs in potency_kwargs.items()
        }
        by_assay: dict[tuple[str, str], list[pd.DataFrame]] = {}
        for name in datasets:
            # A source usually contributes one (kwargs, assay) spec; a qHTS source in ``both``
            # mode contributes two -- the pooled potency onto luminescent_ac50 and the
            # max-response onto the response head -- so the panel carries both qHTS quantities at
            # once. Each spec loads under its own try, so a representation a given deposit cannot
            # supply (the old Veith CYP1A2 deposit has no Max_Response column) drops only that
            # quantity, not the enzyme's other rows.
            if name in with_response and both_mode:
                specs = [
                    (potency_kwargs[name], "luminescent_ac50"),
                    ({"representation": "max_response"}, RESPONSE_ASSAY),
                ]
            else:
                specs = [(extra_kwargs.get(name, {}), assay_by_source[name])]
            for kwargs, assay_override in specs:
                try:
                    loaded = all_loaders[name](
                        enzyme, challenge, dedup_mode=config.data.dedup_mode, **kwargs
                    )
                except Exception as error:  # noqa: BLE001 -- best-effort, but remember it raised
                    logger.info(
                        "no %s (%s) data for %s (%s); skipping", name, assay_override, enzyme, error
                    )
                    raised[name] = str(error)
                    continue
                if loaded.empty:
                    continue
                contributed[name] += len(loaded)
                # Give every source the same columns, so pooling and row-building are uniform: a
                # source without intervals collapses its bounds to the point and is uncensored.
                for column, default in (
                    ("conf_low", loaded["value"]),
                    ("conf_high", loaded["value"]),
                ):
                    if column not in loaded.columns:
                        loaded[column] = default
                if "is_censored" not in loaded.columns:
                    loaded["is_censored"] = False
                # Pool same-chemistry sources together by default; keep them apart (keyed by the
                # source name) only when the model conditions on the source. The response rows
                # live on their own assay, so they never pool with the potency rows regardless.
                source_key = name if source_heads else "_pooled"
                by_assay.setdefault((assay_override, source_key), []).append(loaded)

        for (assay, source_key), gathered in by_assay.items():
            data = pd.concat(gathered, ignore_index=True)
            if len(gathered) > 1:
                data = pool_same_assay(data)
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
                        "data_space": data_space,
                        "source": source_key,
                    }
                )
            )
    # A requested source that realised nothing is a warning, not a silent skip: say whether
    # it failed to parse or simply held no admitted compound.
    for name in datasets:
        if contributed[name] == 0:
            reason = f"parser error: {raised[name]}" if name in raised else "no admitted rows"
            logger.warning(
                "auxiliary source %r was requested but contributed nothing (%s)", name, reason
            )
    if not frames:
        return empty, empty
    out = pd.concat(frames, ignore_index=True)
    if config.auxiliary.row_selection != "none":
        from cypshift.data.selection import reduce_groups

        out = reduce_groups(
            out,
            config.auxiliary.max_rows_per_group,
            config.auxiliary.row_selection,
            retained["SMILES"].tolist(),
            config.data.seed,
            config.auxiliary.row_share,
            config.auxiliary.row_selection_assays,
        )
    response = empty
    if response_feature_route(config):
        # Rows of the response assay train the helper model, not the main model, so they are
        # held out of the table the per-endpoint budget and the model see.
        is_response = out["assay"] == RESPONSE_ASSAY
        response, out = out[is_response], out[~is_response]
    if config.auxiliary.per_endpoint_budget > 0 and not out.empty:
        out = cap_per_endpoint_budget(out, retained, config.auxiliary.per_endpoint_budget)
    # The realised inventory: how many rows actually entered, by source chemistry and
    # censoring, so a run is judged on what it trained on rather than on what it requested.
    inventory = (
        out.assign(censored=out["is_censored"].astype(bool))
        .groupby(["assay", "censored"])["value"]
        .size()
    )
    logger.info(
        "added %d auxiliary rows from %d enzymes; by assay/censored:\n%s",
        len(out),
        out["target"].nunique(),
        inventory.to_string(),
    )
    return out, response


def save_run_bundle(
    bundle_path: Path,
    config: ExperimentConfig,
    evaluation: pd.DataFrame,
    submission: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Write one run's per-compound out-of-fold predictions, truth, resolved config and scores.

    This is what later pairing, blending and calibration need -- the predictions per compound
    on a shared holdout, not just the aggregate score. The files are written in place into
    ``bundle_path`` (not atomically, and the directory is not cleared first), so a re-run with
    a different config can leave mixed artifacts; the caller namespaces the path by revision to
    avoid that. The bundle records the pooled predictions and the full config (so seed, folds
    and every knob are recoverable), but not per-fold or per-member predictions, a data hash,
    or any fitted calibrator/blend parameters -- those must be recorded separately to replay a
    nested postprocessing choice.
    """
    bundle_path.mkdir(parents=True, exist_ok=True)
    truth_columns = ["Molecule_Name"]
    for endpoint in REGRESSION_ENDPOINTS:
        truth_columns += [endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]
    predictions = submission.rename(columns={e: f"{e}_pred" for e in REGRESSION_ENDPOINTS})
    out_of_fold = evaluation[truth_columns].merge(predictions, on="Molecule_Name", validate="1:1")
    out_of_fold.to_parquet(bundle_path / "out_of_fold.parquet", index=False)
    meta = {
        "config": config.to_dict(),
        "scores": scores.round(4).reset_index().to_dict("records"),
    }
    (bundle_path / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Also write the config on its own, so each result carries a standalone, directly
    # reloadable config file beside its predictions and scores.
    (bundle_path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    logger.info("saved run bundle (%d OOF compounds) to %s", len(out_of_fold), bundle_path)


def run(
    config: ExperimentConfig,
    bundle_path: Path | None = None,
    holdout_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, PredictiveDistribution]:
    """Fit the configured model and score it on a holdout.

    By default the holdout is the similarity-matched one that reproduces the blinded test's
    nearest-neighbour distance profile. Pass ``holdout_mask`` (a boolean over the training
    rows, true where a compound is held out) to score a different view instead -- the
    analogue view from :func:`cypshift.data.splits.analogue_holdout`, say -- without changing
    anything else about the fit.

    With ``bundle_path`` set, an immutable run bundle (out-of-fold predictions, config and
    scores) is written there for later pairing, blending and calibration.

    Returns:
        The score table and the predictive distribution behind it.
    """
    config.validate()
    directory = Path(config.data.directory)
    train = load("inhibition", directory)
    test = load("test", directory)

    if holdout_mask is None:
        mask = similarity_matched_holdout(
            train["SMILES"].tolist(),
            test["SMILES"].tolist(),
            holdout_fraction=config.data.holdout_fraction,
            n_bins=config.data.n_similarity_bins,
            n_rounds=config.data.n_matching_rounds,
            seed=config.data.seed,
        )
    else:
        mask = np.asarray(holdout_mask, dtype=bool)
        if len(mask) != len(train):
            raise ValueError(
                f"holdout_mask has {len(mask)} rows but the training set has {len(train)}"
            )
    retained, evaluation = train.loc[~mask], train.loc[mask]
    # Keep the held-out and blinded-test compounds out of the TDI assay_condition so their
    # preincubated
    # measurement cannot leak into training.
    exclude = set(evaluation["Molecule_Name"]) | set(test["Molecule_Name"])
    observations, screen_model = assemble_training_data(config, retained, exclude_names=exclude)
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

    model = build_model(config, screen_model, targets=targets)
    model.fit(observations)
    prediction = model.predict(evaluation["SMILES"].tolist())

    submission = evaluation[["Molecule_Name"]].copy()
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        submission[endpoint] = prediction.mean[:, column]
    scores = score_submission(submission, evaluation)
    _log_r2_decomposition(submission, evaluation)

    # Only score TDI when the preincubated assay_condition was actually trained; a model that merely
    # owns a predict_is_time_dependent method would otherwise report a meaningless MCC from
    # an untrained second assay_condition.
    if config.sources.tdi_assay_condition:
        tdi = score_tdi(model, evaluation, directory)
        if tdi is not None:
            logger.info(
                "TDI track (ranked by MCC):\n%s", tdi[["MCC_mean", "Accuracy_mean"]].to_string()
            )
    if bundle_path is not None:
        save_run_bundle(bundle_path, config, evaluation, submission, scores)
    return scores, prediction


def run_cv(
    config: ExperimentConfig,
    n_folds: int = 5,
    bundle_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Fit and score the configured model by stratified, scaffold-grouped cross-validation.

    Where :func:`run` scores a single similarity-matched holdout, this predicts every
    training compound exactly once, out of fold. The training set is partitioned into
    ``n_folds`` folds that keep each Bemis-Murcko scaffold whole -- so an analog series never
    straddles the train/validation boundary -- and balance endpoint coverage across folds;
    each fold is then predicted by a model trained on the others. The pooled out-of-fold
    predictions are the clean table blending and calibration need: one prediction per
    compound, none of them in-sample.

    With ``bundle_path`` set, the pooled predictions, config and scores are written there.

    Returns:
        The regression score table over the pooled out-of-fold predictions, those
        predictions, and the TDI classification scores when the model can predict them
        (a dose-response curve), otherwise None.
    """
    config.validate()
    directory = Path(config.data.directory)
    train = load("inhibition", directory)
    test = load("test", directory)

    strata = train[list(REGRESSION_ENDPOINTS)].notna().sum(axis=1).to_numpy()
    folds = stratified_scaffold_folds(
        train["SMILES"].tolist(), strata, n_folds=n_folds, seed=config.data.seed
    )

    out_of_fold = []
    tdi_out_of_fold = []
    curves: dict[int, list[dict]] = {}
    for fold in range(n_folds):
        held_out = folds == fold
        retained, evaluation = train.loc[~held_out], train.loc[held_out]
        # Keep this fold's compounds and the blinded test out of the TDI assay_condition, so a
        # preincubated measurement of a validation compound cannot leak into training.
        exclude = set(evaluation["Molecule_Name"]) | set(test["Molecule_Name"])
        observations, screen_model = assemble_training_data(config, retained, exclude_names=exclude)
        scored = list(ISOFORMS)
        extra = [t for t in dict.fromkeys(observations["target"]) if t not in scored]
        targets = tuple(scored + sorted(extra))
        logger.info(
            "fold %d/%d | %d train, %d held out",
            fold + 1,
            n_folds,
            len(retained),
            int(held_out.sum()),
        )

        model = build_model(config, screen_model, targets=targets)
        fit_options: dict = {}
        graph = config.model.graph
        if graph.monitor_fold:
            # The fold's own labelled compounds, evaluated after every epoch; the pruning limit is
            # the reference run's error for this fold at its half-way epoch, plus the tolerance.
            fit_options["monitor"] = build_observations(evaluation)
            if graph.prune_reference:
                reference = reference_error(graph.prune_reference, fold)
                if reference is not None:
                    fit_options["prune_above"] = reference * (1 + graph.prune_tolerance)
        model.fit(observations, **fit_options)
        if graph.monitor_fold:
            curves[fold] = model.curves
        prediction = model.predict(evaluation["SMILES"].tolist())
        fold_predictions = evaluation[["Molecule_Name"]].copy()
        for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
            fold_predictions[endpoint] = prediction.mean[:, column]
        out_of_fold.append(fold_predictions)

        # The TDI assay_condition, out of fold too, but only when the preincubated assay_condition
        # was actually
        # trained (sources.tdi_assay_condition) and the model can predict it. Without the
        # assay_condition trained,
        # the call is meaningless (all-not-TDI, MCC ~0) and just wastes the bootstrap.
        if config.sources.tdi_assay_condition and hasattr(model, "predict_is_time_dependent"):
            flags = model.predict_is_time_dependent(evaluation["SMILES"].tolist())
            fold_tdi = evaluation[["Molecule_Name"]].copy()
            for endpoint in CLASSIFICATION_ENDPOINTS:
                fold_tdi[endpoint] = flags[:, list(ISOFORMS).index(endpoint.split("_")[0])]
            tdi_out_of_fold.append(fold_tdi)

    submission = pd.concat(out_of_fold, ignore_index=True)
    scores = score_submission(submission, train)
    _log_r2_decomposition(submission, train)
    tdi_scores = None
    if tdi_out_of_fold:
        pooled_tdi = pd.concat(tdi_out_of_fold, ignore_index=True)
        tdi_scores = _score_tdi_out_of_fold(pooled_tdi, directory)
        if tdi_scores is not None:
            logger.info(
                "TDI CV (ranked by MCC):\n%s",
                tdi_scores[["MCC_mean", "Accuracy_mean"]].to_string(),
            )
    if bundle_path is not None:
        save_run_bundle(bundle_path, config, train, submission, scores)
        if curves:
            write_curves(bundle_path, curves)
    return scores, submission, tdi_scores


def _score_tdi_out_of_fold(predictions: pd.DataFrame, directory: Path) -> pd.DataFrame | None:
    """Score pooled out-of-fold is_TDI flags against the TDI labels.

    ``predictions`` is a frame of ``Molecule_Name`` plus the two is_TDI columns, pooled from
    every fold. The folds are over the direct-labelled training set, so the predictions -- and
    therefore this MCC -- cover only the direct-labelled compounds that also carry a TDI label,
    not the full TDI label universe: a compound with a TDI label but no direct curve can train
    but never receives an out-of-fold classification prediction here.
    """
    labels = load("tdi", directory)
    shared = predictions.merge(
        labels[["Molecule_Name", *CLASSIFICATION_ENDPOINTS]],
        on="Molecule_Name",
        suffixes=("_pred", ""),
    )
    if shared.empty:
        logger.info("no TDI label overlaps the out-of-fold predictions")
        return None
    predicted = pd.DataFrame({e: shared[f"{e}_pred"] for e in CLASSIFICATION_ENDPOINTS})
    truth = shared[list(CLASSIFICATION_ENDPOINTS)].astype("boolean")
    logger.info("scoring TDI on %d out-of-fold compounds", len(shared))
    return score_classification(predicted, truth)


def _log_r2_decomposition(submission: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Log R² split into its ranking ceiling and the spread and offset it loses.

    Ranking (rho) transfers from this holdout to the blinded test; the spread and offset
    terms do not, since the holdout's potency distribution differs from the test's. The
    split says which part of a change is a real modelling gain rather than a calibration
    the holdout cannot vouch for.

    The predictions are joined to the truth by ``Molecule_Name`` first: under cross-validation
    the pooled submission is in fold order, not the training-table order, so a positional mask
    would pair each prediction with the wrong compound.
    """
    merged = evaluation[["Molecule_Name", *REGRESSION_ENDPOINTS]].merge(
        submission[["Molecule_Name", *REGRESSION_ENDPOINTS]],
        on="Molecule_Name",
        suffixes=("", "_pred"),
        validate="one_to_one",
    )
    rows = {}
    for endpoint in REGRESSION_ENDPOINTS:
        measured = merged[endpoint].notna().to_numpy()
        rows[endpoint.split("_")[0]] = r2_decomposition(
            merged.loc[measured, endpoint].to_numpy(float),
            merged.loc[measured, f"{endpoint}_pred"].to_numpy(float),
        )
    table = pd.DataFrame(rows).T[
        ["rho", "spread_ratio", "offset", "r2", "ranking_ceiling", "spread_loss", "offset_loss"]
    ]
    logger.info("R² decomposition (rho transfers; spread/offset do not):\n%s", table.round(3))
    return table


def score_tdi(model: object, evaluation: pd.DataFrame, directory: Path) -> pd.DataFrame | None:
    """Score the time-dependent inhibition track, when the model can predict it.

    Only a model of the dose-response curve can. The label is defined by comparing the
    preincubated assay_condition's midpoint against the direct assay_condition's, so predicting it
    means
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
