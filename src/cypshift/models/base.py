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
from numpy.typing import NDArray

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


@runtime_checkable
class Model(Protocol):
    """A prediction model: ligand-only, co-folded affinity head, or interaction potential."""

    name: str
    endpoints: tuple[str, ...]

    def fit(self, smiles: list[str], targets: NDArray[np.float64]) -> "Model":
        """Train on SMILES and a ``(n_compounds, n_endpoints)`` target matrix.

        Missing targets are NaN and must be handled by the model, since coverage is
        26-48% per isoform and is not missing at random.
        """
        ...

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict a distribution for each compound and endpoint."""
        ...
