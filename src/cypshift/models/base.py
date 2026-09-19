"""The common interface every prediction model implements.

A model does not return a point estimate. It returns a *predictive distribution*,
because the scored metric has a soft threshold and the point that minimises expected error
under a soft-threshold loss is not the posterior mean. Deciding that point is the job of
:mod:`cypshift.decision`, and it needs a distribution to integrate against.

The distribution is carried as samples rather than as a mean and a standard
deviation. That is deliberate: an active/inactive posterior is bimodal, and a
Gaussian summary cannot represent the shape that makes the decision non-trivial. An
ensemble supplies samples for free along its member axis.
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.data.assembly import TIERS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictiveDistribution:
    """Predicted pIC50 for a set of compounds across a set of endpoints.

    Attributes:
        samples: Shape ``(n_members, n_compounds, n_endpoints)``. Spread along the
            member axis is epistemic: disagreement between models that saw the same
            data. For a single deterministic model this axis has length one.
        endpoints: Column names, matching the last axis of ``samples``.
        aleatoric_variance: Optional, same shape as ``samples``. Per-member predicted
            measurement noise, for models that estimate it. Kept separate so the two
            sources can be combined by the law of total variance rather than conflated.
    """

    samples: NDArray[np.float64]
    endpoints: tuple[str, ...]
    aleatoric_variance: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.samples.ndim != 3:
            raise ValueError(
                f"samples must be (members, compounds, endpoints), got {self.samples.shape}"
            )
        if self.samples.shape[-1] != len(self.endpoints):
            raise ValueError(
                f"{self.samples.shape[-1]} endpoint columns but {len(self.endpoints)} names"
            )
        if (
            self.aleatoric_variance is not None
            and self.aleatoric_variance.shape != self.samples.shape
        ):
            raise ValueError("aleatoric_variance must match the shape of samples")

    @property
    def n_members(self) -> int:
        return self.samples.shape[0]

    @property
    def n_compounds(self) -> int:
        return self.samples.shape[1]

    @property
    def mean(self) -> NDArray[np.float64]:
        """Posterior mean, shape ``(n_compounds, n_endpoints)``."""
        return self.samples.mean(axis=0)

    @property
    def epistemic_variance(self) -> NDArray[np.float64]:
        """Variance across members: the part more data could reduce."""
        return self.samples.var(axis=0)

    @property
    def aleatoric_mean_variance(self) -> NDArray[np.float64]:
        """Mean predicted measurement noise, or zeros if the model does not estimate it."""
        if self.aleatoric_variance is None:
            return np.zeros_like(self.mean)
        return self.aleatoric_variance.mean(axis=0)

    @property
    def total_std(self) -> NDArray[np.float64]:
        """Total predictive standard deviation, by the law of total variance."""
        return np.sqrt(self.epistemic_variance + self.aleatoric_mean_variance)

    def quantile(self, q: float) -> NDArray[np.float64]:
        """Empirical quantile across members, shape ``(n_compounds, n_endpoints)``."""
        return np.quantile(self.samples, q, axis=0)

    def draw(self, n_draws: int, seed: int = 0) -> NDArray[np.float64]:
        """Monte Carlo draws of the predicted value, shape ``(n_draws, compounds, endpoints)``.

        Members are resampled with replacement and, where a model reports aleatoric
        variance, Gaussian noise of that scale is added. This is what the decision
        layer integrates the soft-threshold loss against.
        """
        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, self.n_members, size=n_draws)
        draws = self.samples[chosen]
        if self.aleatoric_variance is not None:
            draws = draws + rng.normal(0.0, np.sqrt(self.aleatoric_variance[chosen]))
        return draws

    def endpoint(self, name: str) -> "PredictiveDistribution":
        """Narrow to a single endpoint, keeping the same structure."""
        index = self.endpoints.index(name)
        variance = (
            None if self.aleatoric_variance is None else self.aleatoric_variance[:, :, [index]]
        )
        return PredictiveDistribution(self.samples[:, :, [index]], (name,), variance)


@dataclass(frozen=True)
class ResolvedStage:
    """One training stage: its data facets and its concrete (filled-in) strategy.

    ``tiers`` selects proximity, ``assays`` selects the data type (``None`` = every assay).
    """

    tiers: tuple[str, ...]
    assays: tuple[str, ...] | None
    epochs: int
    lr: float
    freeze_trunk: bool


def training_stages(
    training: object | None, default_epochs: int, default_lr: float
) -> list[ResolvedStage]:
    """The curriculum as concrete stages, ``None`` epochs/lr filled from the model.

    A model with no training config, or an empty curriculum, trains one pooled stage
    over every tier and assay -- the baseline single fit. Otherwise each configured stage
    keeps its data facets and its swappable strategy, reusing the model's own epochs and
    learning rate wherever the stage left them unset.
    """
    if training is not None and hasattr(training, "effective_stages"):
        stages = training.effective_stages()
    else:
        stages = getattr(training, "stages", None)
    if not stages:
        return [ResolvedStage(TIERS, None, default_epochs, default_lr, False)]
    return [
        ResolvedStage(
            tuple(stage.tiers),
            None if stage.assays is None else tuple(stage.assays),
            default_epochs if stage.epochs is None else stage.epochs,
            default_lr if stage.lr is None else stage.lr,
            bool(stage.freeze_trunk),
        )
        for stage in stages
    ]


def rows_for_stage(observations: pd.DataFrame, stage: ResolvedStage) -> NDArray[np.bool_]:
    """Mask over rows this stage trains on: tier in ``tiers`` and assay in ``assays``.

    A missing ``tier`` column means every row qualifies (a run that never tagged tiers);
    ``assays`` of ``None`` places no constraint on the data type.
    """
    if "tier" not in observations:
        return np.ones(len(observations), dtype=bool)
    mask = observations["tier"].isin(stage.tiers).to_numpy()
    if stage.assays is not None:
        mask = mask & observations["assay"].isin(stage.assays).to_numpy()
    return mask


def active_stages(observations: pd.DataFrame, stages: list[ResolvedStage]) -> list[ResolvedStage]:
    """Drop stages with no rows this run; fall back to one pooled stage if all are empty.

    A curriculum can name a facet a given run does not populate -- ``distant_aux`` when no
    auxiliary source is on, or the TDI assay when its arm is off. Such stages are skipped
    rather than training on nothing, and if every stage is empty the model still trains
    once over whatever is present, so a run never produces an unfitted model.
    """
    present = [stage for stage in stages if rows_for_stage(observations, stage).any()]
    if present:
        return present
    first = stages[0]
    return [ResolvedStage(TIERS, None, first.epochs, first.lr, False)]


@runtime_checkable
class Model(Protocol):
    """A prediction model: ligand-only, co-folded affinity head, or interaction potential."""

    name: str
    endpoints: tuple[str, ...]

    def fit(self, observations: pd.DataFrame) -> "Model":
        """Train on the long observation table built by
        :func:`cypshift.data.assembly.build_observations`.

        One row per (compound, enzyme, assay) measurement, carrying ``value``,
        ``conf_low``, ``conf_high``, ``weight`` and ``is_censored``. Long rather than a
        wide target matrix because coverage is 26-48% per isoform and not missing at
        random, and because a censored screen reading is a different kind of
        observation from a fitted midpoint rather than a missing one.
        """
        ...

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict a distribution for each compound and endpoint."""
        ...
