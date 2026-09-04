"""Training on the metric's own loss rather than on squared error.

The challenge charges the distance from a prediction to the nearest bound of the
true value's credible interval, and nothing at all inside it. Squared error does not
resemble that. A compound already predicted inside its band still produces gradient
under mean squared error, so the model keeps refining predictions that are already
scored perfect, while the dead-zone loss has exactly zero gradient there and spends
capacity only on compounds that are actually being charged.

It also absorbs censored labels without special handling. A single-concentration
screen reading outside the calibration range establishes only that a compound is
inactive -- a one-sided band running down from a bound. Under this loss, predicting
anywhere below that bound costs nothing, which is precisely what the reading
supports. The 2,855 censored CYP2D6 pseudo-labels become ordinary training rows.

Uncertainty comes from dropout left active at prediction time rather than from
reseeding. Members that differ only by seed were measured on this dataset to
disagree by 0.03 log units while being wrong by 0.45, so their spread carries
almost no information about where a prediction is unreliable.
"""

import logging

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import ASSAYS, conditioning_matrix
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.conditioned import SCORED_ASSAY

logger = logging.getLogger(__name__)

#: Stands in for an unbounded side of a censored label.
UNBOUNDED = 1e4


def dead_zone_loss(
    prediction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """The scored loss, as a differentiable objective.

    Mirrors ``rae_soft_threshold_absolute_error``'s numerator: distance to the nearest
    bound, zero inside. Only the numerator, because the denominator is a property of
    the truth and does not depend on the prediction, so it cannot affect the gradient.

    Args:
        prediction: Predicted pIC50, shape ``(n,)``.
        lower: Lower bound of each label's band. Use a large negative value for a
            label censored from below.
        upper: Upper bound of each label's band.
        weight: Per-row trust, so a screen pseudo-label counts for less than a curve.

    Returns:
        Weighted mean loss, a scalar.
    """
    above = torch.clamp(prediction - upper, min=0.0)
    below = torch.clamp(lower - prediction, min=0.0)
    return ((above + below) * weight).sum() / weight.sum().clamp(min=1e-8)


class ConditionedMLP(nn.Module):
    """Molecular features concatenated with the target and assay one-hot block."""

    def __init__(self, n_features: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x).squeeze(-1)


class DeadZoneRoute:
    """Structure-free route trained on the challenge's own loss."""

    name = "structure_free_deadzone"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        n_bits: int = 2048,
        hidden: int = 256,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        n_forward_passes: int = 30,
        loss: str = "dead_zone",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.n_bits = n_bits
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_forward_passes = n_forward_passes
        self.loss = loss
        self.seed = seed
        self._model: ConditionedMLP | None = None
        self._centre: NDArray[np.float32] | None = None
        self._scale: NDArray[np.float32] | None = None

    def _features(self, smiles: list[str], context: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.hstack([featurise(smiles, n_bits=self.n_bits), context]).astype(np.float32)

    def _standardise(self, features: NDArray[np.float32], fit: bool) -> torch.Tensor:
        if fit:
            self._centre = features.mean(axis=0)
            self._scale = features.std(axis=0) + 1e-6
        return torch.from_numpy((features - self._centre) / self._scale)

    def fit(self, observations: pd.DataFrame) -> "DeadZoneRoute":
        """Train on the observation table, using each row's own band.

        Rows without explicit bounds are treated as points, which reduces the loss to
        plain absolute error for them.
        """
        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        context = conditioning_matrix(usable, self.targets, self.assays)
        features = self._standardise(self._features(usable["SMILES"].tolist(), context), fit=True)
        value = usable["value"].to_numpy(float)
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -UNBOUNDED, lower)

        lower_t = torch.tensor(lower, dtype=torch.float32)
        upper_t = torch.tensor(upper, dtype=torch.float32)
        value_t = torch.tensor(value, dtype=torch.float32)
        weight_t = torch.tensor(usable["weight"].to_numpy(float), dtype=torch.float32)

        self._model = ConditionedMLP(features.shape[1], self.hidden, self.dropout)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        n = len(usable)
        logger.info("training %s loss on %d observations", self.loss, n)

        for epoch in range(self.epochs):
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                index = order[start : start + self.batch_size]
                optimiser.zero_grad()
                predicted = self._model(features[index])
                if self.loss == "dead_zone":
                    batch_loss = dead_zone_loss(
                        predicted, lower_t[index], upper_t[index], weight_t[index]
                    )
                else:
                    squared = (predicted - value_t[index]) ** 2 * weight_t[index]
                    batch_loss = squared.sum() / weight_t[index].sum().clamp(min=1e-8)
                batch_loss.backward()
                optimiser.step()
                total += batch_loss.detach().item() * len(index)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d  loss %.4f", epoch, total / n)
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict with dropout left on, so repeated passes give a genuine spread."""
        if self._model is None:
            raise RuntimeError("route is not fitted")
        self._model.train()  # keep dropout active: this is the uncertainty source
        n = len(smiles)
        samples = np.empty((self.n_forward_passes, n, len(self.endpoints)))
        with torch.no_grad():
            for column, endpoint in enumerate(self.endpoints):
                query = pd.DataFrame(
                    {"target": [endpoint.split("_")[0]] * n, "assay": [SCORED_ASSAY] * n}
                )
                context = conditioning_matrix(query, self.targets, self.assays)
                features = self._standardise(self._features(smiles, context), fit=False)
                for pass_index in range(self.n_forward_passes):
                    samples[pass_index, :, column] = self._model(features).numpy()
        return PredictiveDistribution(samples, self.endpoints)
