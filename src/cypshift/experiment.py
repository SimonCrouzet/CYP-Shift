"""Experiment configuration: versioned defaults in a file, overridable per run.

Everything an experiment can vary lives here rather than in a function signature, so a
run is reproducible from a single JSON file and a sweep is a matter of listing values.
Choices the literature does not settle are exposed as options rather than picked and
defended after the fact -- notably how the assay condition is encoded and how the
cheap primary screen is combined with the dose-response data, where the published
evidence points in a different direction from the obvious implementation.
"""

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

from cypshift.config import ISOFORMS
from cypshift.data.assembly import ASSAYS, DATA_SPACES
from cypshift.data.auxiliary import AUXILIARY_WEIGHTING as _AUXILIARY_WEIGHTING
from cypshift.data.conditioning import DESCRIPTOR_SOURCES as _DESCRIPTOR_SOURCES
from cypshift.models.tdi import TDI_FLOOR_DEFAULT, TDI_SHIFT_DEFAULT

#: Isoforms eligible for a per-isoform adaptation block, plus the ``"none"`` default.
ADAPT_TARGETS = ("none", *ISOFORMS)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.json")

#: How the assay condition and target enter the model.
#:
#: How the target and assay are encoded as input columns, **for the baseline models
#: only**.
#:
#: These exist because a model that predicts a pIC50 directly has nowhere to put the
#: measurement channel, so it has to be told which assay a row came from. The curve
#: model does not need them: screen-versus-curve lives in its likelihood and
#: direct-versus-preincubated is structural. See :data:`CURVE_CONDITIONING` for the
#: axis that applies to it.
#:
#: A flat encoding over (target, assay) pairs was removed. It shared nothing between
#: conditions, and once the assay dimension moved out of conditioning there was no
#: argument left for it.
CONDITIONING_STRATEGIES = (
    "factorised",
    "metadata",
    "embedding",
    "descriptor",
    "heads",
    "film",
    "calibration",
)

#: How the curve model makes its output depend on the enzyme. This is a *structural*
#: choice rather than an encoding, which is why it is a separate axis.
#:
#: ``heads`` gives one output head per (enzyme, assay_condition) -- maximally flexible, but a head
#: is an index, so an enzyme outside the scored set trains parameters that are then
#: discarded and nothing pretrains. ``descriptor`` uses one shared head given a vector
#: describing the enzyme's pocket, which is less free per enzyme but lets an auxiliary
#: enzyme's data improve the scored ones. Which wins is measured, not assumed.
CURVE_CONDITIONING = ("heads", "descriptor", "film")

#: Re-exported so a config can be validated without importing the data package.
DESCRIPTOR_SOURCES = _DESCRIPTOR_SOURCES

#: Re-exported for the same reason.
AUXILIARY_WEIGHTING = _AUXILIARY_WEIGHTING

#: How the single-concentration screen is folded in when ``sources.screen`` is on.
#: ``extra_rows`` pools calibrated pseudo-labels (or, for the curve, the raw reading) onto
#: the potency scale; ``predicted_label``/``embedding`` fit a screen_model feature column.
#: ``response_head`` passes the screen reading as an inhibited fraction to the graph's
#: response head (assay ``SCREEN_RESPONSE_ASSAY``) instead of mapping it to a pIC50 through
#: the isotonic calibrator; graph only.
SCREEN_STRATEGIES = ("extra_rows", "predicted_label", "embedding", "response_head")

#: Molecular foundation-model embeddings that can be appended as a feature block. Each
#: (but ``none``) needs a precomputed cache from scripts/compute_embeddings.py.
EMBEDDING_SOURCES = ("none", "mammal")

MODEL_KINDS = ("boosted", "neural", "graph", "curve", "per_target_ensemble", "tfm")
OBJECTIVES = ("squared", "absolute", "soft_threshold", "expectile", "pinball")

#: How a graph ensemble's members differ. ``reseed`` varies only the initialisation seed;
#: ``bagging`` also trains each member on a seeded bootstrap resample, so members see the
#: data differently -- the diversity a reseed ensemble lacks.
ENSEMBLE_STRATEGIES = ("reseed", "bagging")

#: Pretrained D-MPNN encoder to initialise the graph model from. ``none`` trains from
#: scratch; ``chemeleon`` loads the chemprop-native CheMeleon foundation encoder;
#: ``mammal`` initialises from the cached MAMMAL embedding as a fixed input block.
GRAPH_PRETRAINING = ("none", "chemeleon", "mammal")

#: How the Veith qHTS panel is represented: a pooled pIC50, or a 0-1 max-response on its own
#: assay (which the graph consumes through a separate head).
QHTS_REPRESENTATIONS = ("pic50", "max_response")

#: How auxiliary rows are matched against the challenge set for dedup: exact canonical SMILES,
#: or the coarser InChIKey connectivity block.
DEDUP_MODES = ("canonical_smiles", "inchikey_block")


@dataclass
class DataConfig:
    directory: str = "data/raw"
    holdout_fraction: float = 0.2
    n_similarity_bins: int = 8
    n_matching_rounds: int = 3
    seed: int = 0
    #: How auxiliary rows are matched against the challenge set to drop overlaps: exact
    #: ``canonical_smiles`` (default) or the coarser ``inchikey_block`` (InChIKey connectivity
    #: block, collapsing salts/stereoisomers). See cypshift.data.chembl.dedup_key. Swept.
    dedup_mode: str = "canonical_smiles"


@dataclass
class FeatureConfig:
    #: 4096 rather than 2048: measured -0.0110 macro ST-RAE over three splits, the
    #: only hyperparameter change that helped. Leaves, trees and learning rate were
    #: all already at or past their optimum.
    fingerprint_bits: int = 4096
    fingerprint_radius: int = 2
    #: Include the Morgan fingerprint block. On for every model by default; a sweepable
    #: toggle so a dense-only feature set (physchem/mechanism/ionization/embedding, no
    #: fingerprint) can be tried -- it is what TabPFN and other dense learners prefer. The
    #: graph ignores it (it learns its own representation from the molecular graph).
    use_fingerprint: bool = True
    use_physicochemical: bool = True
    use_mechanistic: bool = True
    #: Append the ionization / through-space pharmacophore block: charge state at pH 7.4
    #: (dimorphite-dl) and a seeded-conformer 3D distance from a protonated nitrogen to the
    #: nearest aromatic ring. Computed and cached (:mod:`cypshift.features.ionization`);
    #: off by default and swept, since the conformer step is the slow one.
    use_ionization_geometry: bool = False
    #: Where the computed ionization/geometry features are cached (one parquet, keyed by
    #: canonical SMILES).
    ionization_cache: str = "data/external/ionization/features.parquet"
    #: Append Boltz-2 cofold features (affinity + pose/interaction) per (compound,
    #: enzyme). Off by default: it needs a precomputed cofold cache
    #: (scripts/cofold_structures.py); missing pairs are NaN. Toggled by
    #: ``sources.structure``; this is only where the cofold cache lives. See
    #: :mod:`cypshift.structure`.
    structure_cache: str = "data/structure"
    #: Append a molecular foundation-model embedding as a feature block. ``none`` uses
    #: only the fingerprint/descriptor features; ``mammal`` appends cached MAMMAL
    #: embeddings (see EMBEDDING_SOURCES). The vectors are precomputed offline by
    #: scripts/compute_embeddings.py into data/external/embeddings and only read here,
    #: so the main environment never depends on the embedding model.
    molecular_embedding: str = "none"


@dataclass
class AuxiliaryConfig:
    """Which enzymes to draw auxiliary rows for, and how much to trust each.

    Enzymes sit on a single binding-site distance axis: a scored enzyme is at distance 0
    from itself (its own rows from an outside source), a relative further out. Distance
    is a mean Z-scale difference per aligned position, on a scale where two unrelated
    sites score 5.27 and a conservative Leu-to-Ile swap scores 0.62. See
    :mod:`cypshift.data.auxiliary`. Auxiliary rows appear only when a source dataset
    (chembl/sair/octant/veith) is on -- a default run trains on the challenge data alone.
    """

    #: The one enzyme-admission knob. Every enzyme within this binding-site distance of a
    #: scored one is drawn from the active source datasets. ``0.0`` admits only the scored
    #: enzymes' own rows (distance 0, the CYP2D6 lever); ``1.5`` adds seven human CYPs;
    #: ``3.0`` adds seventeen, most of them past halfway to the unrelated-site distance.
    max_distance: float = 1.5
    #: Weight on the scored enzymes' own (distance-0) rows. Kept separate from the
    #: distance weighting because these are the same target measured by a different,
    #: noisier assay -- worth down-weighting even though the distance is zero.
    scored_weight: float = 1.0
    #: How a relative's weight falls off with distance; see :data:`AUXILIARY_WEIGHTING`.
    weighting: str = "exponential"
    #: Distance at which "inverse" halves and "exponential" falls to 1/e.
    scale: float = 1.0
    #: Multiplies a relative's distance weight. Lower it when auxiliary volume swamps the
    #: target: CYP2C19 alone brings about 17,000 Veith rows against roughly 6,000
    #: dose-response labels across all four scored enzymes.
    target_weight: float = 1.0
    #: Keep a qHTS panel's inactive compounds as censored low-potency rows (``pIC50`` below
    #: the panel's detection limit) rather than dropping them. The Veith deposit is mostly
    #: inactive -- 6,822 against 1,776 active on CYP2D6 -- and keeping only the actives
    #: trains on a population biased toward hits, which the blinded test is not. Swept.
    include_qhts_inactives: bool = True
    #: Keep the Tox21 P450-Glo re-screen's inactive compounds as censored rows at the assay's
    #: highest tested concentration rather than dropping them (the loader keeps only actives by
    #: default). Off by default; swept.
    tox21_include_inactives: bool = False
    #: Rescale each scored endpoint's near-auxiliary rows so their summed weight equals this
    #: fraction of that endpoint's challenge-label weight. ``0.0`` leaves the distance/source
    #: weights untouched (the default); ``0.5`` caps a scored enzyme's near-auxiliary rows at
    #: half its challenge weight. Applies to the scored enzymes' own outside rows (near_aux);
    #: relatives (distant_aux) keep the distance weighting. See pipeline.cap_per_endpoint_budget.
    per_endpoint_budget: float = 0.0


@dataclass
class SourcesConfig:
    """Which evidence to use, each an independent yes/no.

    Deliberately just switches. *How* each source is folded in is the separate strategy
    axis: the screen through ``multifidelity.screen_strategy`` and ``screen_weight``,
    auxiliary data through ``auxiliary.max_distance`` / ``weighting`` / ``scale`` and the
    weight fields, structure through ``features.structure_cache``. Turning several on at
    once is expected -- they compose.
    """

    #: The single-concentration screen (17,504 censored readings).
    screen: bool = True
    #: The preincubated (TDI) assay_condition -- needed for the classification track, costly for the
    #: regression one, so off by default.
    tdi_assay_condition: bool = False
    #: ChEMBL auxiliary potencies.
    chembl: bool = False
    #: SAIR auxiliary potencies (needs data/external/sair.parquet).
    sair: bool = False
    #: Octant/OpenADMET CYP3A4 potencies -- the challenge's own assay platform, with
    #: real credible intervals (needs data/external/octant/inhibition.tsv).
    octant: bool = False
    #: Veith qHTS panel potencies (PubChem). A luminescent ranking signal and the same
    #: campaign as ChEMBL's AC50 block, so off by default; downloads and caches on use.
    veith: bool = False
    #: How Veith qHTS enters: as a pooled ``pic50`` (default) or as a 0-1 ``max_response`` on
    #: its own assay. A data-side choice, not gated on the model -- but max_response needs a
    #: model with a separate head for it, so validation requires model.kind=graph + veith.
    qhts_representation: str = "pic50"
    #: 2025 NCATS Tox21 P450-Glo re-screen potencies (PubChem). Biochemical, all four
    #: scored CYPs, a distinct library from Veith; off by default, downloads on use.
    tox21: bool = False
    #: Novartis reversible pIC50 for CYP3A4/2C9/2D6 (Polaris). ~16k rows per enzyme, a
    #: large independent CYP2D6 lever; off by default, needs the Polaris SDK.
    novartis: bool = False
    #: Boltz-2 cofold features (affinity + heme-coordination pose), from the cache.
    structure: bool = False


@dataclass
class ConditioningConfig:
    #: How a model is told which enzyme and assay a row came from. Not every strategy
    #: suits every model -- see :func:`cypshift.pipeline.supported_conditioning`:
    #: ``factorised``/``metadata`` are one-hot input blocks (all input models);
    #: ``embedding`` is a learned table, ``descriptor`` appends the pocket vector,
    #: ``heads`` gives one output head per condition, ``film`` lets the descriptor scale
    #: the trunk (the last three are the neural model's, mirroring the curve model's).
    strategy: str = "factorised"
    #: Width of the learned table when strategy is ``embedding``.
    embedding_dim: int = 8
    #: Which pocket-descriptor sources build the vector for ``descriptor`` and ``film``;
    #: see :data:`cypshift.data.conditioning.DESCRIPTOR_SOURCES`.
    descriptor_sources: tuple[str, ...] = ("curated",)


@dataclass
class MultifidelityConfig:
    #: Whether to train on the preincubated assay_condition as well as the direct one.
    #:
    #: Off by default, which is a choice about which leaderboard a run is aimed at
    #: rather than a judgement that the assay_condition is useless. Including it is what makes the
    #: classification track work at all -- without it the curve model's second assay_condition
    #: trains on nothing and predicts "no" for every compound, MCC 0.0 -- but it costs
    #: 0.033 to 0.061 macro ST-RAE on the regression track at every weight tried, more
    #: than every improvement measured so far put together.
    #:
    #: How the single-concentration screen is folded in when ``sources.screen`` is on:
    #: extra_rows (pooled points / pseudo-labels), predicted_label or embedding (a
    #: screen_model feature). The "use it at all" decision is ``sources.screen``.
    screen_strategy: str = "extra_rows"
    screen_weight: float = 0.3
    tdi_weight: float = 0.5
    #: Hidden-layer width of the screen screen_model model (an MLP), which is also the
    #: dimension of the ``embedding`` screen_model. Wired into fit_screen_model; swept.
    screen_model_hidden: int = 16


@dataclass
class BoostedConfig:
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.6


@dataclass
class NeuralConfig:
    hidden: int = 256
    dropout: float = 0.2
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    forward_passes: int = 30
    #: is_TDI decision thresholds for its direct-vs-preincubated assay_condition comparison; our
    #: swept
    #: heuristic (cypshift.models.tdi), not a challenge constant. Needs sources.tdi_assay_condition.
    tdi_shift_threshold: float = TDI_SHIFT_DEFAULT
    tdi_assay_floor: float = TDI_FLOOR_DEFAULT


@dataclass
class CurveConfig:
    """The dose-response curve model, which predicts curve parameters per assay_condition."""

    hidden: int = 256
    dropout: float = 0.2
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    forward_passes: int = 30
    #: ``heads`` or ``descriptor``; see :data:`CURVE_CONDITIONING`.
    conditioning: str = "heads"
    #: Which descriptor sources build the vector when ``conditioning`` is
    #: ``descriptor``; see :data:`cypshift.data.conditioning.DESCRIPTOR_SOURCES`.
    #: Defaults to the curated table alone, since eight enzymes will not support a
    #: wide vector without leave-one-target-out evidence that it transfers.
    descriptor_sources: tuple[str, ...] = ("curated",)
    #: Half-width of the interval placed around a single-well screen reading, as an
    #: inhibited fraction. One measurement with no replicate does not support a tight
    #: bound, and stating a tighter one would let the screen overrule fitted curves.
    screen_noise: float = 0.1
    #: The model's is_TDI decision thresholds on its predicted assay_condition shift; our heuristic,
    #: not
    #: a challenge constant (see cypshift.models.tdi). Swept, since the best threshold on
    #: noisy predicted shifts need not equal the one that defined the labels.
    tdi_shift_threshold: float = TDI_SHIFT_DEFAULT
    tdi_assay_floor: float = TDI_FLOOR_DEFAULT
    #: Asymmetry of the dose-response dead-zone loss; >0.5 penalises under-prediction more
    #: (errs high), the curve's analogue of the expectile/pinball objectives. Swept.
    loss_tau: float = 0.5


@dataclass
class GraphConfig:
    hidden: int = 300
    depth: int = 3
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    #: Dataloader worker processes (0 = main process). Higher keeps the GPU fed and speeds
    #: training; the right value is machine-dependent, so it is set per run, not swept.
    num_workers: int = 0
    #: How ensemble members differ; see :data:`ENSEMBLE_STRATEGIES`. Swept.
    ensemble: str = "reseed"
    #: A pretrained D-MPNN encoder to initialise from; see :data:`GRAPH_PRETRAINING`. Swept.
    pretraining: str = "none"
    #: Where the CheMeleon encoder is cached (downloaded on first use).
    pretraining_cache: str = "data/external/chemeleon/chemeleon_mp.pt"
    #: Fine-tune the pretrained encoder (default) or freeze it as a fixed featuriser so only
    #: the readout trains. Only meaningful with a pretrained encoder. Swept.
    pretraining_finetune: bool = True
    #: Train one encoder per scored isoform (separated trunks) instead of one shared trunk.
    #: Removes cross-isoform interference -- CYP2D6 does not share a representation with the
    #: lipophilicity-driven isoforms -- at the cost of one training per isoform. Swept.
    separate_trunks: bool = False
    #: Condition on the row's source dataset (keeping same-chemistry sources apart rather than
    #: pooling them), so each source gets its own calibration on the shared trunk. Swept.
    source_heads: bool = False
    #: One scored isoform to give an isoform-specific adaptation block on top of the shared
    #: readout, or ``"none"``. A residual MLP on the pooled representation whose output is added
    #: to the value prediction, gated by that isoform's target one-hot so it is zero for every
    #: other isoform; the encoder and the rest of the readout stay shared. Rejected with
    #: ``separate_trunks``. Swept. See :class:`cypshift.models.graph._TargetAdaptedRegressionFFN`.
    adapt_target: str = "none"
    #: Hidden width of the per-isoform adaptation block; ignored when ``adapt_target`` is
    #: ``"none"``.
    adapt_hidden: int = 64
    #: Hidden width of the readout FFN, set independently of the message-passing width
    #: (``hidden``). ``0`` keeps chemprop's default (300). The reference recipes use a wide FFN
    #: (e.g. 1024) over a narrower encoder, which ``hidden`` alone cannot express. Swept.
    ffn_hidden: int = 0
    #: is_TDI decision thresholds for predict_is_time_dependent, which compares the graph's
    #: direct and preincubated (tdi_drc) assay_conditions. Default to the organisers' rule
    #: (config.py);
    #: swept. Only used when sources.tdi_assay_condition trained the preincubated assay_condition.
    tdi_shift_threshold: float = TDI_SHIFT_DEFAULT
    tdi_assay_floor: float = TDI_FLOOR_DEFAULT


@dataclass
class TfmConfig:
    """The tabular foundation model family; see :data:`TFM_BACKENDS`."""

    #: Which tabular foundation model backs it (``tabpfn``; Mitra to come). Swept.
    backend: str = "tabpfn"
    #: PCA target width for the dimensionality adaptation that fits the shared feature set to
    #: TabPFN's ~2000-feature ceiling (and off its weak spot, raw sparse fingerprints). Swept.
    pca_components: int = 500


TFM_BACKENDS = ("tabpfn",)


@dataclass
class ModelConfig:
    kind: str = "boosted"
    objective: str = "soft_threshold"
    #: Asymmetry level for the ``expectile``/``pinball`` objectives; >0.5 errs high, which the
    #: soft-threshold metric rewards. Ignored by the other objectives. Swept.
    objective_tau: float = 0.5
    #: Weight on a squared ``(pred - midpoint)**2`` term added to the soft-threshold loss on
    #: uncensored rows (midpoint = (lower + upper) / 2). ``0.0`` leaves the dead-zone loss
    #: untouched. Requires ``objective='soft_threshold'``; graph model only. Swept.
    point_loss_weight: float = 0.0
    #: Weight on a within-batch pairwise hinge over value-task rows whose credible intervals are
    #: disjoint: a pair ordered against its intervals is penalised by ``max(0, margin - gap)``,
    #: weighted by the weaker row's observation weight. ``0.0`` is off. Requires
    #: ``objective='soft_threshold'``; graph model only. Swept.
    ranking_loss_weight: float = 0.0
    n_members: int = 5
    boosted: BoostedConfig = field(default_factory=BoostedConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)
    tfm: TfmConfig = field(default_factory=TfmConfig)


@dataclass
class Stage:
    """One stage of the training curriculum: which data, and how to train on it.

    Two facets select the data, after the pattern that separates target relatedness from
    assay context: ``data_spaces`` is the proximity facet (a subset of
    :data:`cypshift.data.assembly.DATA_SPACES`) and ``assays`` the data-type facet (a subset of
    :data:`cypshift.data.assembly.ASSAYS`, or ``None`` for every assay). Both are fixed by
    the curriculum. The rest are the *swappable strategy* for the stage: ``epochs`` and
    ``lr`` (``None`` reuses the model's own), and ``freeze_trunk``, which holds the trunk
    fixed so only the conditioning heads adapt (neural/curve; the boosted tree has no
    trunk and instead continues boosting from the previous stage). The same stage can be
    run frozen or fully fine-tuned without changing which data it trains on.
    """

    data_spaces: tuple[str, ...] = DATA_SPACES
    assays: tuple[str, ...] | None = None
    epochs: int | None = None
    lr: float | None = None
    freeze_trunk: bool = False


_DRC = "direct_inhibition_drc"
_SCREEN = "direct_inhibition_screen"

#: Named curricula, so the whole stage sequence is one swept categorical rather than a
#: hand-edited list. Each is a *way of fine-tuning*, not a different dataset -- turn on
#: the sources you want and pick a preset. ``pooled`` is the baseline single fit.
CURRICULUM_PRESETS: dict[str, list[Stage]] = {
    "pooled": [Stage()],
    # Coarse to fine: pretrain on outside data, adapt to the challenge, sharpen on the
    # scored dose-response with the trunk frozen.
    "coarse_to_fine": [
        Stage(data_spaces=("distant_aux", "near_aux")),
        Stage(data_spaces=("challenge",), assays=(_SCREEN, _DRC), lr=5e-4),
        Stage(data_spaces=("challenge",), assays=(_DRC,), epochs=25, lr=2e-4, freeze_trunk=True),
    ],
    # The single-concentration screen as its own stage before the dose-response.
    "assay_first": [
        Stage(data_spaces=("distant_aux", "near_aux")),
        Stage(data_spaces=("challenge",), assays=(_SCREEN,), lr=5e-4),
        Stage(data_spaces=("challenge",), assays=(_DRC,), epochs=25, lr=2e-4, freeze_trunk=True),
    ],
    # Coarse to fine, then a short unfrozen polish at a very low lr (progressive unfreezing).
    "progressive": [
        Stage(data_spaces=("distant_aux", "near_aux")),
        Stage(data_spaces=("challenge",), assays=(_SCREEN, _DRC), lr=5e-4),
        Stage(data_spaces=("challenge",), assays=(_DRC,), epochs=25, lr=2e-4, freeze_trunk=True),
        Stage(data_spaces=("challenge",), assays=(_DRC,), epochs=10, lr=5e-5),
    ],
}


@dataclass
class TrainingConfig:
    """The training curriculum: an ordered list of stages, trained farthest data_space first.

    Two ways to set it. A ``preset`` names a whole curriculum from
    :data:`CURRICULUM_PRESETS`, so the fine-tuning strategy is one swept categorical --
    the presets are alternative *ways of fine-tuning*, not different datasets. Or leave
    ``preset`` unset and give an explicit ``stages`` list. The default is the pooled
    baseline: one fit over the whole weighted table. The stages (which data, in what
    order) are set deliberately; what a stage's strategy varies is ``freeze_trunk``,
    ``lr`` and ``epochs``. Every model runs the same sequence with its own transition
    mechanism.
    """

    preset: str | None = None
    stages: list[Stage] = field(default_factory=lambda: [Stage()])
    #: Rescale each scored endpoint's rows so the four contribute equal total weight, which
    #: approximates the macro-averaged metric the leaderboard uses rather than letting the
    #: data-rich endpoints dominate a pooled fit. Effective now that the boosted objective
    #: applies row weights. Swept, off by default.
    macro_balance: bool = False

    def effective_stages(self) -> list[Stage]:
        """The curriculum to run: the named preset if set, otherwise the explicit stages."""
        if self.preset is not None:
            return CURRICULUM_PRESETS[self.preset]
        return self.stages


@dataclass
class TrackingConfig:
    enabled: bool = False
    project: str = "cyp-shift"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Everything one run varies."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    multifidelity: MultifidelityConfig = field(default_factory=MultifidelityConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> "ExperimentConfig":
        """Reject an unusable configuration before anything expensive runs."""
        checks = [
            (self.conditioning.strategy, CONDITIONING_STRATEGIES, "conditioning.strategy"),
            (self.model.curve.conditioning, CURVE_CONDITIONING, "model.curve.conditioning"),
            *(
                (source, DESCRIPTOR_SOURCES, "model.curve.descriptor_sources")
                for source in self.model.curve.descriptor_sources
            ),
            (self.auxiliary.weighting, AUXILIARY_WEIGHTING, "auxiliary.weighting"),
            (
                self.multifidelity.screen_strategy,
                SCREEN_STRATEGIES,
                "multifidelity.screen_strategy",
            ),
            (self.model.kind, MODEL_KINDS, "model.kind"),
            (self.model.objective, OBJECTIVES, "model.objective"),
            (self.features.molecular_embedding, EMBEDDING_SOURCES, "features.molecular_embedding"),
            (self.model.graph.ensemble, ENSEMBLE_STRATEGIES, "model.graph.ensemble"),
            (self.model.graph.pretraining, GRAPH_PRETRAINING, "model.graph.pretraining"),
            (self.sources.qhts_representation, QHTS_REPRESENTATIONS, "sources.qhts_representation"),
            (self.data.dedup_mode, DEDUP_MODES, "data.dedup_mode"),
            (self.model.graph.adapt_target, ADAPT_TARGETS, "model.graph.adapt_target"),
        ]
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed}")
        if self.sources.qhts_representation == "max_response" and not (
            self.model.kind == "graph" and self.sources.veith
        ):
            raise ValueError(
                "sources.qhts_representation='max_response' needs a model that can consume it: "
                "model.kind='graph' with sources.veith on"
            )
        if self.multifidelity.screen_strategy == "response_head":
            if self.model.kind != "graph":
                raise ValueError(
                    "multifidelity.screen_strategy='response_head' needs a model with a response "
                    f"head: model.kind='graph', not {self.model.kind!r}"
                )
            if self.sources.qhts_representation == "max_response":
                # Both would want the graph's single response column; keeping them mutually
                # exclusive avoids conflating a qHTS max-response with a screen fraction on
                # one head (two heads is a later experiment, not this one).
                raise ValueError(
                    "multifidelity.screen_strategy='response_head' and "
                    "sources.qhts_representation='max_response' both claim the graph response "
                    "head; enable only one"
                )
        if not 0 < self.data.holdout_fraction < 1:
            raise ValueError("data.holdout_fraction must lie strictly between 0 and 1")
        if self.model.n_members < 1:
            raise ValueError("model.n_members must be at least 1")
        if self.training.preset is not None and self.training.preset not in CURRICULUM_PRESETS:
            raise ValueError(
                f"training.preset={self.training.preset!r} is not one of "
                f"{tuple(CURRICULUM_PRESETS)}"
            )
        stages = self.training.effective_stages()
        if not stages:
            raise ValueError("training must resolve to at least one stage")
        for index, stage in enumerate(stages):
            if not stage.data_spaces:
                raise ValueError(f"training.stages[{index}].data_spaces is empty")
            unknown_data_spaces = set(stage.data_spaces) - set(DATA_SPACES)
            if unknown_data_spaces:
                raise ValueError(
                    f"training.stages[{index}].data_spaces has unknown "
                    f"{sorted(unknown_data_spaces)}"
                )
            unknown_assays = set(stage.assays or ()) - set(ASSAYS)
            if unknown_assays:
                raise ValueError(
                    f"training.stages[{index}].assays has unknown {sorted(unknown_assays)}"
                )
        self._reject_silently_ignored()
        return self

    def _reject_silently_ignored(self) -> None:
        """Reject settings a run would accept then ignore, so a sweep cannot credit a
        configuration with testing something it never applied."""
        for name, weight in (
            ("auxiliary.scored_weight", self.auxiliary.scored_weight),
            ("auxiliary.target_weight", self.auxiliary.target_weight),
            ("multifidelity.screen_weight", self.multifidelity.screen_weight),
            ("auxiliary.per_endpoint_budget", self.auxiliary.per_endpoint_budget),
            ("model.point_loss_weight", self.model.point_loss_weight),
            ("model.ranking_loss_weight", self.model.ranking_loss_weight),
        ):
            if weight < 0:
                raise ValueError(f"{name} must be >= 0, got {weight!r}")
        if self.sources.structure and self.model.kind != "boosted":
            raise ValueError(
                f"sources.structure is wired only into the boosted model, not {self.model.kind!r}"
            )
        # The point and ranking terms augment the graph's own chemprop loss; no other model
        # reads them, so a non-graph run carrying a non-zero weight would be credited with an
        # experiment it never ran.
        if self.model.kind != "graph":
            for name, weight in (
                ("model.point_loss_weight", self.model.point_loss_weight),
                ("model.ranking_loss_weight", self.model.ranking_loss_weight),
            ):
                if weight > 0:
                    raise ValueError(
                        f"{name} is wired only into the graph model, not {self.model.kind!r}"
                    )
            if self.model.graph.adapt_target != "none":
                raise ValueError(
                    "model.graph.adapt_target is wired only into the graph model, not "
                    f"{self.model.kind!r}"
                )
        if self.model.kind == "graph":
            if self.features.molecular_embedding != "none":
                raise ValueError(
                    "the graph model reads embeddings through model.graph.pretraining='mammal', "
                    "not features.molecular_embedding"
                )
            stages = self.training.effective_stages()
            if len(stages) > 1:
                raise ValueError("the graph model does not run a multi-stage training curriculum")
            if stages:
                # The graph factory never passes the stage to GraphNetwork, whose fit runs its
                # own epochs over the whole assembled table. A stage that restricts the data
                # (a data_space subset or an assay filter) or sets a per-stage strategy (epochs,
                # lr, freeze_trunk) is therefore silently ignored, so reject it rather than
                # credit a run with a curriculum it never applied. The trivial pooled stage --
                # every data_space, every assay, the model's own schedule -- is what the graph
                # actually runs, so it is accepted (compared as sets, since a loaded config
                # carries data_spaces as a list).
                stage = stages[0]
                restricts_data = set(stage.data_spaces) != set(DATA_SPACES) or stage.assays
                sets_strategy = (
                    stage.epochs is not None or stage.lr is not None or stage.freeze_trunk
                )
                if restricts_data or sets_strategy:
                    raise ValueError(
                        "the graph model ignores per-stage training settings (data_spaces, "
                        "assays, epochs, lr, freeze_trunk); configure training through the "
                        "model.graph.* fields instead"
                    )
            if self.model.graph.separate_trunks and self.model.graph.adapt_target != "none":
                # Separate trunks already give each isoform its own encoder, so a per-isoform
                # adapter on top is redundant and its gate (a target one-hot) is absent.
                raise ValueError(
                    "model.graph.adapt_target needs the shared trunk; it is meaningless with "
                    "model.graph.separate_trunks=True"
                )
            point_or_rank = self.model.point_loss_weight > 0 or self.model.ranking_loss_weight > 0
            if point_or_rank and self.model.objective != "soft_threshold":
                # The point and ranking terms layer onto the soft-threshold loss only; _build
                # ignores them under the other objectives, which would credit a run with a
                # term it never applied.
                raise ValueError(
                    "model.point_loss_weight / ranking_loss_weight apply only to "
                    f"model.objective='soft_threshold', not {self.model.objective!r}"
                )
        if self.model.kind == "curve" and self.model.objective != "soft_threshold":
            raise ValueError(
                "the curve model uses its own dose-response loss; model.objective="
                f"{self.model.objective!r} would be ignored"
            )
        if self.model.kind == "tfm":
            # The tabular backend has no sample weights, censoring or objective, and trains only
            # on the uncensored reference-assay rows (see TabularFoundationModel.fit). Reject the
            # settings that would be accepted and then ignored or dropped.
            if self.model.objective != "soft_threshold":
                raise ValueError(
                    "model.kind='tfm' has no training objective; model.objective is ignored"
                )
            if self.training.macro_balance or self.training.preset is not None:
                raise ValueError(
                    "model.kind='tfm' ignores sample weights and the training curriculum"
                )
            if self.sources.tdi_assay_condition:
                raise ValueError("model.kind='tfm' cannot use the preincubated assay_condition")
            auxiliary_on = [
                name
                for name in ("chembl", "sair", "octant", "veith", "tox21", "novartis")
                if getattr(self.sources, name)
            ]
            if auxiliary_on:
                raise ValueError(
                    "model.kind='tfm' trains only on reference-assay rows; the auxiliary "
                    f"sources {auxiliary_on} would be dropped"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        """Dotted keys, which is the shape sweep tools and trackers expect."""
        flat: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(f"{prefix}.{key}" if prefix else key, item)
            else:
                flat[prefix] = value

        walk("", self.to_dict())
        return flat

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        def build(target: type, source: dict[str, Any]) -> Any:
            kwargs = {}
            for item in fields(target):
                if item.name not in source:
                    continue
                given = source[item.name]
                if is_dataclass(item.type) and isinstance(given, dict):
                    kwargs[item.name] = build(item.type, given)
                elif get_origin(item.type) is list and isinstance(given, list):
                    (element,) = get_args(item.type)
                    kwargs[item.name] = (
                        [build(element, one) for one in given] if is_dataclass(element) else given
                    )
                else:
                    kwargs[item.name] = given
            return target(**kwargs)

        return build(cls, values).validate()

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH, **overrides: Any) -> "ExperimentConfig":
        """Load defaults from a file, then apply dotted overrides.

        Args:
            path: JSON file holding the versioned defaults.
            **overrides: Dotted keys such as ``model__kind="neural"``, using double
                underscores because a dot is not valid in a keyword argument.
        """
        values = json.loads(path.read_text()) if path.exists() else {}
        config = cls.from_dict(values)
        for dotted, value in overrides.items():
            if value is None:
                continue
            config.override(dotted.replace("__", "."), value)
        return config.validate()

    def override(self, dotted: str, value: Any) -> None:
        """Set one nested field, e.g. ``"model.boosted.num_leaves"``."""
        target: Any = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"no such configuration section: {dotted}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"no such configuration field: {dotted}")
        setattr(target, parts[-1], value)
        logger.info("override %s = %r", dotted, value)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path
