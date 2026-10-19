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
from cypshift.data.assembly import (
    ASSAYS,
    REFERENCE_ASSAY,
    condition_index,
    encode_conditions,
    n_conditions,
)
from cypshift.data.conditioning import descriptor_conditions
from cypshift.features.assembly import molecular_feature_matrix
from cypshift.features.scaling import BlockScaler
from cypshift.models.base import (
    PredictiveDistribution,
    active_stages,
    rows_for_stage,
    training_stages,
)
from cypshift.models.shared import SCORED_ASSAY
from cypshift.models.tdi import TDI_FLOOR_DEFAULT, TDI_SHIFT_DEFAULT

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


OBJECTIVES = ("squared", "absolute", "soft_threshold", "expectile", "pinball")


def _objective(
    name: str,
    prediction: torch.Tensor,
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Weighted mean of the requested per-row objective.

    ``expectile`` and ``pinball`` take ``tau``: at ``tau>0.5`` they penalise under-prediction
    more, the upward bias the soft-threshold metric rewards, while keeping a gradient
    everywhere (unlike the dead-zone loss) so the predicted spread does not collapse.
    """
    if name == "soft_threshold":
        return soft_threshold_loss(prediction, lower, upper, weight)
    if name == "squared":
        per_row = (prediction - value) ** 2
    elif name == "absolute":
        per_row = (prediction - value).abs()
    elif name == "expectile":
        side = torch.where(value > prediction, tau, 1.0 - tau)
        per_row = side * (value - prediction) ** 2
    elif name == "pinball":
        residual = value - prediction
        per_row = torch.maximum(tau * residual, (tau - 1.0) * residual)
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


class EmbeddedMLP(nn.Module):
    """Molecular features concatenated with a learned (target, assay) embedding.

    The one-hot conditioning of :class:`ConditionedMLP` spends a column per level and
    lets the first trunk layer learn its own projection. This instead looks the
    condition up in a small trainable table, which shares parameters across rows of the
    same enzyme and gives a fixed-width, swept bottleneck (``embedding_dim``) rather
    than a width tied to the number of levels.
    """

    def __init__(
        self,
        n_features: int,
        n_conditions: int,
        embedding_dim: int = 8,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_conditions, embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(n_features + embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([x, self.embedding(index)], dim=1)).squeeze(-1)


class HeadsMLP(nn.Module):
    """A shared trunk with one output head per (target, assay) condition.

    The scalar analogue of the curve model's ``heads`` mode: the trunk is common to
    every enzyme, and a final linear layer holds one column per condition, of which the
    row's own condition is read off. Maximally flexible per condition, but a condition
    with no training rows never trains its column.
    """

    def __init__(
        self, n_features: int, n_conditions: int, hidden: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.Linear(hidden // 2, n_conditions)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        alled = self.heads(self.trunk(x))
        return alled.gather(1, index.unsqueeze(1)).squeeze(1)


class FilmMLP(nn.Module):
    """The pocket descriptor scales and shifts the trunk (feature-wise modulation).

    The scalar analogue of the curve model's ``film`` mode. A small layer turns the
    descriptor-and-assay vector into a per-feature scale and shift applied to the shared
    representation, so the enzyme multiplies the molecular features rather than only
    adding a constant offset. Initialised to the identity, so training starts from the
    unconditioned model.
    """

    def __init__(
        self, n_features: int, n_condition: int, hidden: int = 256, dropout: float = 0.2
    ) -> None:
        super().__init__()
        width = hidden // 2
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.ReLU(),
        )
        self.modulation = nn.Linear(n_condition, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(x)
        scale, shift = self.modulation(condition).chunk(2, dim=1)
        modulated = self.dropout((1.0 + scale) * hidden + shift)
        return self.head(modulated).squeeze(-1)


class CalibratedMLP(nn.Module):
    """A shared potency conditioned on the enzyme, plus a per-assay affine calibration.

    The trunk sees the molecule and the *target* one-hot -- not the assay -- and predicts
    one latent potency ``f(molecule, enzyme)`` on the reference assay's scale. A learned
    per-assay bias is then added, recentred every forward pass so the reference assay's
    bias is exactly zero. Because predictions are made at the reference assay, the
    systematic offset of each auxiliary assay (a luminescent AC50 sitting below a
    fluorogenic IC50, say) is absorbed into that assay's bias and shapes ``f`` without
    dragging its scale -- the proteochemometric fix for pooling measurements from
    different platforms. A slope is deliberately omitted: the offset is the dominant
    systematic effect and a bias-only calibration stays identifiable.
    """

    def __init__(
        self,
        n_features: int,
        n_assays: int,
        reference_index: int,
        hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.reference_index = reference_index
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.bias = nn.Embedding(n_assays, 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, assay_index: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x).squeeze(-1)
        # Recentre so the reference assay's bias is zero, keeping the shared potency on
        # the reference scale regardless of what the other assays' offsets drift to.
        bias = (self.bias.weight - self.bias.weight[self.reference_index]).squeeze(-1)
        return shared + bias[assay_index]


class NeuralNetwork:
    """Structure-free model trained on the challenge's own loss."""

    name = "proteochemometric_neural"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        conditioning: str = "factorised",
        n_bits: int = 2048,
        physchem: bool = True,
        mechanism: bool = True,
        fingerprint_radius: int = 2,
        screen_model: object | None = None,
        hidden: int = 256,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        n_forward_passes: int = 30,
        loss: str = "soft_threshold",
        tau: float = 0.5,
        tdi_shift_threshold: float = TDI_SHIFT_DEFAULT,
        tdi_assay_floor: float = TDI_FLOOR_DEFAULT,
        embedding_dim: int = 8,
        descriptor_sources: tuple[str, ...] = ("curated",),
        seed: int = 0,
        training: object | None = None,
        molecular_embedding: str = "none",
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.conditioning = conditioning
        self.n_bits = n_bits
        #: Which molecular feature blocks to include and the fingerprint radius.
        #: These are read by featurise; toggling physchem changes the dense-block width,
        #: which the scaler follows. Swept via features.* in the config.
        self.physchem = physchem
        self.mechanism = mechanism
        self.fingerprint_radius = fingerprint_radius
        #: Optional callable appended to the feature matrix, used to feed the
        #: single-concentration screen in as an input rather than as extra training
        #: rows. See :mod:`cypshift.data.screen_model`.
        self.screen_model = screen_model
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_forward_passes = n_forward_passes
        self.loss = loss
        self.tau = tau
        self.tdi_shift_threshold = tdi_shift_threshold
        self.tdi_assay_floor = tdi_assay_floor
        #: Width of the learned (target, assay) table when conditioning == "embedding".
        self.embedding_dim = embedding_dim
        #: Pocket-descriptor sources for the "descriptor" and "film" strategies.
        self.descriptor_sources = descriptor_sources
        self.seed = seed
        #: The training curriculum (an object exposing ``.stages``), or None for one
        #: pooled stage over every data_space -- the baseline.
        self.training = training
        #: Which cached foundation-model embedding to append, or "none".
        self.molecular_embedding = molecular_embedding
        #: Append the computed ionization / conformer-geometry block, and where it caches.
        self.ionization_geometry = ionization_geometry
        self.ionization_cache = ionization_cache
        self._model: nn.Module | None = None
        self._scaler: BlockScaler | None = None

    def _features(
        self, smiles: list[str], one_hot: NDArray[np.float32] | None = None
    ) -> NDArray[np.float32]:
        molecular = molecular_feature_matrix(
            smiles,
            n_bits=self.n_bits,
            fingerprint_radius=self.fingerprint_radius,
            physchem=self.physchem,
            mechanism=self.mechanism,
            screen_model=self.screen_model,
            ionization_geometry=self.ionization_geometry,
            ionization_cache=self.ionization_cache,
            molecular_embedding=self.molecular_embedding,
        )
        if one_hot is None:
            return molecular.astype(np.float32)
        return np.hstack([molecular, one_hot]).astype(np.float32)

    def _encode(
        self, frame: pd.DataFrame, smiles: list[str], fit: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scaled molecular features and, for the embedding strategy, condition indices.

        The molecular block is scaled on its own and the (sparse one-hot or low-cardinality
        descriptor) condition is appended raw afterwards, so standardising never fills in
        the condition's zeros. Returns ``(features, index)`` where ``index`` is None unless
        conditioning is ``embedding``/``heads``, which look the condition up in a learned
        table rather than feeding it as columns.
        """
        molecular = self._scale_blocks(self._features(smiles), fit=fit)
        if self.conditioning == "calibration":
            # The trunk conditions on the enzyme (target one-hot in the features); the
            # assay is not an input but an index into the per-assay calibration bias.
            target_hot = torch.tensor(
                [[t == name for name in self.targets] for t in frame["target"]],
                dtype=torch.float32,
            )
            assay_index = torch.tensor(
                [self.assays.index(a) for a in frame["assay"]], dtype=torch.long
            )
            return torch.cat([molecular, target_hot], dim=1), assay_index
        if self.conditioning in ("embedding", "heads"):
            index = torch.tensor(
                condition_index(frame, self.targets, self.assays), dtype=torch.long
            )
            return molecular, index
        if self.conditioning == "film":
            block = descriptor_conditions(frame, self.assays, self.descriptor_sources)
            return molecular, torch.tensor(block, dtype=torch.float32)
        if self.conditioning == "descriptor":
            block = torch.tensor(
                descriptor_conditions(frame, self.assays, self.descriptor_sources),
                dtype=torch.float32,
            )
            return torch.cat([molecular, block], dim=1), None
        one_hot = torch.tensor(
            encode_conditions(frame, self.conditioning, self.targets, self.assays),
            dtype=torch.float32,
        )
        return torch.cat([molecular, one_hot], dim=1), None

    def _scale_blocks(self, molecular: NDArray[np.float32], fit: bool) -> torch.Tensor:
        """Scale the molecular block: log1p the fingerprint counts, standardise every dense
        descriptor after it -- physchem, mechanistic, screen_model, ionization and embedding --
        rather than physchem alone, which left the appended continuous blocks unscaled."""
        if fit:
            self._scaler = BlockScaler(
                n_count_columns=self.n_bits, n_dense_columns=molecular.shape[1] - self.n_bits
            )
            self._scaler.fit(molecular)
        if self._scaler is None:
            raise RuntimeError("model is not fitted")
        return torch.from_numpy(self._scaler.transform(molecular))

    def fit(self, observations: pd.DataFrame) -> "NeuralNetwork":
        """Train on the observation table, using each row's own interval.

        Rows without explicit bounds are treated as points, which reduces the loss to
        plain absolute error for them.
        """
        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        features, condition = self._encode(usable, usable["SMILES"].tolist(), fit=True)
        value = usable["value"].to_numpy(float)
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -UNBOUNDED, lower)

        lower_t = torch.tensor(lower, dtype=torch.float32)
        upper_t = torch.tensor(upper, dtype=torch.float32)
        value_t = torch.tensor(value, dtype=torch.float32)
        weight_t = torch.tensor(usable["weight"].to_numpy(float), dtype=torch.float32)

        n_features = features.shape[1]
        n_cond = n_conditions(self.targets, self.assays)
        if self.conditioning == "embedding":
            self._model = EmbeddedMLP(
                n_features, n_cond, self.embedding_dim, self.hidden, self.dropout
            )
        elif self.conditioning == "heads":
            self._model = HeadsMLP(n_features, n_cond, self.hidden, self.dropout)
        elif self.conditioning == "film":
            self._model = FilmMLP(n_features, condition.shape[1], self.hidden, self.dropout)
        elif self.conditioning == "calibration":
            self._model = CalibratedMLP(
                n_features,
                len(self.assays),
                self.assays.index(REFERENCE_ASSAY),
                self.hidden,
                self.dropout,
            )
        else:
            self._model = ConditionedMLP(n_features, self.hidden, self.dropout)

        def optimise(rows: torch.Tensor, epochs: int, lr: float, freeze_trunk: bool) -> None:
            """Train the current model on a subset of rows, carrying its weights forward."""
            if len(rows) == 0:
                return
            if freeze_trunk:
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(False)
            trainable = [p for p in self._model.parameters() if p.requires_grad]
            if not trainable:
                # A headless conditioning (factorised/metadata) is all trunk, so freezing
                # it leaves nothing to fine-tune. Skip rather than crash; the pretrained
                # weights stand.
                logger.info("stage frozen with no trainable head; skipping")
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(True)
                return
            optimiser = torch.optim.Adam(trainable, lr=lr)
            m = len(rows)
            for epoch in range(epochs):
                order = rows[torch.randperm(m)]
                total = 0.0
                for start in range(0, m, self.batch_size):
                    index = order[start : start + self.batch_size]
                    optimiser.zero_grad()
                    batch = features[index]
                    predicted = (
                        self._model(batch, condition[index])
                        if condition is not None
                        else self._model(batch)
                    )
                    batch_loss = _objective(
                        self.loss,
                        predicted,
                        value_t[index],
                        lower_t[index],
                        upper_t[index],
                        weight_t[index],
                        self.tau,
                    )
                    batch_loss.backward()
                    optimiser.step()
                    total += batch_loss.detach().item() * len(index)
                if epoch % 20 == 0 or epoch == epochs - 1:
                    logger.info("  epoch %2d  loss %.4f", epoch, total / m)
            if freeze_trunk:
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(True)

        all_rows = torch.arange(len(usable))
        stages = training_stages(self.training, self.epochs, self.learning_rate)
        for stage in active_stages(usable, stages):
            mask = rows_for_stage(usable, stage)
            rows = all_rows[torch.tensor(mask)]
            logger.info(
                "stage data_spaces=%s assays=%s: %d rows, %d epochs, lr %.1e%s",
                stage.data_spaces,
                stage.assays or "all",
                int(mask.sum()),
                stage.epochs,
                stage.lr,
                " (trunk frozen)" if stage.freeze_trunk else "",
            )
            optimise(rows, stage.epochs, stage.lr, stage.freeze_trunk)
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
                features, condition = self._encode(query, smiles, fit=False)
                for pass_index in range(self.n_forward_passes):
                    predicted = (
                        self._model(features, condition)
                        if condition is not None
                        else self._model(features)
                    )
                    samples[pass_index, :, column] = predicted.numpy()
        return PredictiveDistribution(samples, self.endpoints)

    def _predict_pic50_at(self, smiles: list[str], enzyme: str, assay: str) -> NDArray[np.float64]:
        """Dropout-mean pIC50 for one enzyme under one assay condition."""
        self._model.train()
        query = pd.DataFrame(
            {"target": [enzyme] * len(smiles), "assay": [assay] * len(smiles), "SMILES": smiles}
        )
        features, condition = self._encode(query, smiles, fit=False)
        passes = []
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                out = (
                    self._model(features, condition)
                    if condition is not None
                    else self._model(features)
                )
                passes.append(out.numpy())
        return np.mean(passes, axis=0)

    def predict_is_time_dependent(self, smiles: list[str]) -> NDArray[np.bool_]:
        """TDI call by comparing the direct and preincubated assay_conditions, with swept
        thresholds.

        Requires the preincubated assay_condition in training (``sources.tdi_assay_condition``). One
        column per
        isoform, matching the curve and graph models' shape.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted")
        floor, shift = self.tdi_assay_floor, self.tdi_shift_threshold
        flags = []
        for enzyme in (e.split("_")[0] for e in REGRESSION_ENDPOINTS):
            direct = self._predict_pic50_at(smiles, enzyme, "direct_inhibition_drc")
            preincubated = self._predict_pic50_at(smiles, enzyme, "tdi_drc")
            flags.append(
                np.where(
                    direct > floor,
                    (preincubated - direct) > shift,
                    preincubated > floor + shift,
                )
            )
        return np.stack(flags, axis=1)
