"""A feed-forward network over molecular features, with the objective as a parameter.

The training objective is swappable and is meant to be swept, not assumed. Squared
error is the default everywhere; the challenge scores *soft-threshold absolute
error*, which charges the distance to the nearest credible-interval bound and nothing
inside it. Those are different shapes -- under squared error a compound already
predicted inside its interval still produces gradient, so the model keeps refining
predictions already scored perfect, while under the soft-threshold objective the
gradient there is exactly zero and capacity goes to compounds actually being charged.

Which objective generalises best is an empirical question, so ``loss`` is an argument
and every run reports every metric regardless of what it trained on.

It also absorbs censored labels without special handling. A single-concentration
screen reading outside the calibration range establishes only that a compound is
inactive -- a one-sided interval running down from a bound. Under this loss, predicting
anywhere below that bound costs nothing, which is precisely what the reading
supports. The 2,855 censored CYP2D6 pseudo-labels become ordinary training rows.

Feature scaling is per-kind rather than uniform, which matters here in a way it does
not for a tree ensemble: see :mod:`cypshift.features.scaling`. Z-scoring the sparse
fingerprint block alongside the dense descriptors destroys its sparsity and measurably
costs accuracy.

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
from cypshift.data.assembly import ASSAYS, encode_conditions
from cypshift.features.molecular import featurise, physicochemical
from cypshift.features.scaling import BlockScaler
from cypshift.models.base import PredictiveDistribution
from cypshift.models.shared import SCORED_ASSAY

logger = logging.getLogger(__name__)

#: Stands in for an unbounded side of a censored label.
UNBOUNDED = 1e4


def soft_threshold_loss(
    prediction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Soft-threshold absolute error, as a differentiable objective.

    This is the numerator of the challenge's ``rae_soft_threshold_absolute_error``:
    distance to the nearest bound, zero inside. Only the numerator, because the
    denominator is a property of the truth and cannot affect the gradient.

    Args:
        prediction: Predicted pIC50, shape ``(n,)``.
        lower: Lower bound of each label's interval. Use a large negative value for a
            label censored from below.
        upper: Upper bound of each label's interval.
        weight: Per-row trust, so a screen pseudo-label counts for less than a curve.

    Returns:
        Weighted mean loss, a scalar.
    """
    above = torch.clamp(prediction - upper, min=0.0)
    below = torch.clamp(lower - prediction, min=0.0)
    return ((above + below) * weight).sum() / weight.sum().clamp(min=1e-8)


OBJECTIVES = ("squared", "absolute", "soft_threshold")


def _objective(
    name: str,
    prediction: torch.Tensor,
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean of the requested per-row objective."""
    if name == "soft_threshold":
        return soft_threshold_loss(prediction, lower, upper, weight)
    if name == "squared":
        per_row = (prediction - value) ** 2
    elif name == "absolute":
        per_row = (prediction - value).abs()
    else:
        raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")
    return (per_row * weight).sum() / weight.sum().clamp(min=1e-8)


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


class NeuralNetwork:
    """Structure-free model trained on the challenge's own loss."""

    name = "structure_free_neural"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        conditioning: str = "factorised",
        n_bits: int = 2048,
        hidden: int = 256,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        n_forward_passes: int = 30,
        loss: str = "soft_threshold",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.conditioning = conditioning
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
        self._scaler: BlockScaler | None = None

    def _features(self, smiles: list[str], one_hot: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.hstack([featurise(smiles, n_bits=self.n_bits), one_hot]).astype(np.float32)

    def _scale_blocks(self, features: NDArray[np.float32], fit: bool) -> torch.Tensor:
        """Apply per-kind scaling: log1p counts, standardise descriptors, pass the rest."""
        if fit:
            n_dense = len(physicochemical(["CCO"]).columns)
            self._scaler = BlockScaler(n_count_columns=self.n_bits, n_dense_columns=n_dense)
            self._scaler.fit(features)
        if self._scaler is None:
            raise RuntimeError("model is not fitted")
        return torch.from_numpy(self._scaler.transform(features))

    def fit(self, observations: pd.DataFrame) -> "NeuralNetwork":
        """Train on the observation table, using each row's own interval.

        Rows without explicit bounds are treated as points, which reduces the loss to
        plain absolute error for them.
        """
        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        one_hot = encode_conditions(usable, self.conditioning, self.targets, self.assays)
        features = self._scale_blocks(self._features(usable["SMILES"].tolist(), one_hot), fit=True)
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
                batch_loss = _objective(
                    self.loss,
                    predicted,
                    value_t[index],
                    lower_t[index],
                    upper_t[index],
                    weight_t[index],
                )
                batch_loss.backward()
                optimiser.step()
                total += batch_loss.detach().item() * len(index)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d  loss %.4f", epoch, total / n)
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict with dropout left on, so repeated passes give a genuine spread."""
        if self._model is None:
            raise RuntimeError("model is not fitted")
        self._model.train()  # keep dropout active: this is the uncertainty source
        n = len(smiles)
        samples = np.empty((self.n_forward_passes, n, len(self.endpoints)))
        with torch.no_grad():
            for column, endpoint in enumerate(self.endpoints):
                query = pd.DataFrame(
                    {"target": [endpoint.split("_")[0]] * n, "assay": [SCORED_ASSAY] * n}
                )
                one_hot = encode_conditions(query, self.conditioning, self.targets, self.assays)
                features = self._scale_blocks(self._features(smiles, one_hot), fit=False)
                for pass_index in range(self.n_forward_passes):
                    samples[pass_index, :, column] = self._model(features).numpy()
        return PredictiveDistribution(samples, self.endpoints)
