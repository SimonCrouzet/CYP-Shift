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
from typing import Any

from cypshift.data.auxiliary import AUXILIARY_WEIGHTING as _AUXILIARY_WEIGHTING
from cypshift.data.conditioning import DESCRIPTOR_SOURCES as _DESCRIPTOR_SOURCES

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
CONDITIONING_STRATEGIES = ("factorised", "metadata", "embedding", "descriptor", "heads", "film")

#: How the curve model makes its output depend on the enzyme. This is a *structural*
#: choice rather than an encoding, which is why it is a separate axis.
#:
#: ``heads`` gives one output head per (enzyme, arm) -- maximally flexible, but a head
#: is an index, so an enzyme outside the scored set trains parameters that are then
#: discarded and nothing pretrains. ``descriptor`` uses one shared head given a vector
#: describing the enzyme's pocket, which is less free per enzyme but lets an auxiliary
#: enzyme's data improve the scored ones. Which wins is measured, not assumed.
CURVE_CONDITIONING = ("heads", "descriptor", "film")

#: Re-exported so a config can be validated without importing the data package.
DESCRIPTOR_SOURCES = _DESCRIPTOR_SOURCES

#: Re-exported for the same reason.
AUXILIARY_WEIGHTING = _AUXILIARY_WEIGHTING

MULTIFIDELITY_STRATEGIES = ("none", "extra_rows", "predicted_label", "embedding")

MODEL_KINDS = ("boosted", "neural", "graph", "curve", "per_target_ensemble")
OBJECTIVES = ("squared", "absolute", "soft_threshold")


@dataclass
class DataConfig:
    directory: str = "data/raw"
    holdout_fraction: float = 0.2
    n_similarity_bins: int = 8
    n_matching_rounds: int = 3
    seed: int = 0


@dataclass
class FeatureConfig:
    #: 4096 rather than 2048: measured -0.0110 macro ST-RAE over three splits, the
    #: only hyperparameter change that helped. Leaves, trees and learning rate were
    #: all already at or past their optimum.
    fingerprint_bits: int = 4096
    fingerprint_radius: int = 2
    use_physicochemical: bool = True
    use_mechanistic: bool = True
    #: Append Boltz-2 cofold features (affinity + pose/interaction) per (compound,
    #: enzyme). Off by default: it needs a precomputed cofold cache
    #: (scripts/cofold_structures.py); missing pairs are NaN. See
    #: :mod:`cypshift.structure`.
    use_structure: bool = False
    structure_cache: str = "data/structure"


@dataclass
class AuxiliaryConfig:
    """Which enzymes outside the scored four to train on, and how much to trust them.

    Distance is a mean Z-scale difference per aligned binding-site position, on a scale
    where two unrelated sites score 5.27 and a conservative Leu-to-Ile swap scores 0.62.
    See :mod:`cypshift.data.auxiliary`.
    """

    #: Off by default. Pulling auxiliary data needs a network fetch (ChEMBL) or a warm
    #: cache and pulls thousands of extra rows, so it is opt-in and the sweep turns it
    #: on -- a default run trains on the challenge data alone.
    enabled: bool = False
    #: Where auxiliary rows come from: "chembl" (per-target API pull) or "sair" (the
    #: SandboxAQ cofolded IC50 repository, ~61k rows across the four CYPs; needs the
    #: local index at data/external/sair.parquet).
    source: str = "chembl"
    #: Admit a candidate whose binding site is within this of a scored enzyme. The
    #: default admits seven human CYPs; 3.0 admits seventeen, most of them past halfway
    #: to the distance two unrelated sites score.
    max_distance: float = 1.5
    #: How weight falls off with distance; see :data:`AUXILIARY_WEIGHTING`.
    weighting: str = "exponential"
    #: Distance at which "inverse" halves and "exponential" falls to 1/e.
    scale: float = 1.0
    #: Weight given to a scored enzyme's own rows. Raise it when auxiliary volume
    #: swamps the target: CYP2C19 alone brings about 17,000 Veith rows against roughly
    #: 6,000 dose-response labels across all four scored enzymes.
    target_weight: float = 1.0


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
    #: Whether to train on the preincubated arm as well as the direct one.
    #:
    #: Off by default, which is a choice about which leaderboard a run is aimed at
    #: rather than a judgement that the arm is useless. Including it is what makes the
    #: classification track work at all -- without it the curve model's second arm
    #: trains on nothing and predicts "no" for every compound, MCC 0.0 -- but it costs
    #: 0.033 to 0.061 macro ST-RAE on the regression track at every weight tried, more
    #: than every improvement measured so far put together.
    #:
    #: There is no weight that avoids this: 0.1 / 0.3 / 0.6 / 1.0 gave regression
    #: 0.780 / 0.801 / 0.781 / 0.773 against 0.740 without the arm, while macro MCC
    #: moved only between 0.14 and 0.19. So the two tracks want different models, and
    #: since they are scored on separate leaderboards there is no reason to compromise:
    #: submit one run with this off and another with it on.
    use_tdi_arm: bool = False
    strategy: str = "extra_rows"
    screen_weight: float = 0.3
    tdi_weight: float = 0.5
    #: Hidden-layer width of the screen surrogate model (an MLP), which is also the
    #: dimension of the ``embedding`` surrogate. Wired into fit_surrogate; swept.
    surrogate_hidden: int = 16


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


@dataclass
class CurveConfig:
    """The dose-response curve model, which predicts curve parameters per arm."""

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


@dataclass
class GraphConfig:
    hidden: int = 300
    depth: int = 3
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64


@dataclass
class ModelConfig:
    kind: str = "boosted"
    objective: str = "soft_threshold"
    n_members: int = 5
    boosted: BoostedConfig = field(default_factory=BoostedConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    curve: CurveConfig = field(default_factory=CurveConfig)


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
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    multifidelity: MultifidelityConfig = field(default_factory=MultifidelityConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
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
            (self.multifidelity.strategy, MULTIFIDELITY_STRATEGIES, "multifidelity.strategy"),
            (self.model.kind, MODEL_KINDS, "model.kind"),
            (self.model.objective, OBJECTIVES, "model.objective"),
        ]
        for value, allowed, name in checks:
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {allowed}")
        if not 0 < self.data.holdout_fraction < 1:
            raise ValueError("data.holdout_fraction must lie strictly between 0 and 1")
        if self.model.n_members < 1:
            raise ValueError("model.n_members must be at least 1")
        return self

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
                kwargs[item.name] = (
                    build(item.type, given)
                    if is_dataclass(item.type) and isinstance(given, dict)
                    else given
                )
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
