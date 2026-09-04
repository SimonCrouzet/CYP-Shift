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

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.json")

#: How the assay condition and target enter the model.
#:
#: ``flat`` gives one column per (target, assay) pair, so nothing is shared between
#: CYP3A4-preincubated and CYP2D6-preincubated. ``factorised`` splits it into a target
#: block and a condition block, letting "preincubation" be one direction shared across
#: every enzyme, which is more sample-efficient with a 4x3 grid. ``metadata`` adds
#: scalar properties of the assay itself, so the encoding describes the experiment
#: rather than merely indexing it.
CONDITIONING_STRATEGIES = ("flat", "factorised", "metadata")

#: How the single-concentration screen is combined with the dose-response curves.
#:
#: ``none`` ignores it. ``extra_rows`` adds calibrated pseudo-labels as further
#: training rows -- the obvious approach, and the one a large multi-fidelity benchmark
#: found loses high-fidelity signal when the cheap data dominates. ``predicted_label``
#: and ``embedding`` instead train a model on the screen and feed its output or its
#: learned representation in as a feature, which is what that benchmark found wins.
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
    fingerprint_bits: int = 2048
    fingerprint_radius: int = 2
    use_physicochemical: bool = True
    use_mechanistic: bool = True


@dataclass
class ConditioningConfig:
    strategy: str = "factorised"
    embedding_dim: int = 8


@dataclass
class MultifidelityConfig:
    strategy: str = "extra_rows"
    screen_weight: float = 0.3
    tdi_weight: float = 0.5
    surrogate_estimators: int = 300


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
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> "ExperimentConfig":
        """Reject an unusable configuration before anything expensive runs."""
        checks = [
            (self.conditioning.strategy, CONDITIONING_STRATEGIES, "conditioning.strategy"),
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
