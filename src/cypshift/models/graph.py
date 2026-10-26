"""A directed message-passing network that learns its own molecular representation.

Everything so far has used a fixed representation -- Morgan fingerprints and computed
descriptors -- and only varied what sits on top. A D-MPNN instead learns the
representation from the molecular graph, which is the standard escape from the
fingerprint's fixed vocabulary of substructures and is what the second-placed entry in
this challenge's public leaderboard uses.

Two things had to be worked out to make Chemprop carry this problem.

**Conditioning.** Chemprop's ``x_d`` slot holds molecule-level features concatenated
to the learned representation *after* aggregation, which is exactly where a
conditioning vector belongs: the graph encoder stays target-agnostic and shared across
every isoform, and the readout sees which enzyme and which assay the row came from.
One row per (compound, target, assay), a single output, and the graph encoder is
trained on every measurement rather than a quarter of them.

**Bounds.** Chemprop passes only labels and predictions to a loss, with no channel for
a per-row credible interval. The workaround, which Chemprop's own
``NLogProbEnrichment`` uses, is to widen the label matrix and slice it inside the loss:
column 0 is the value, 1 the lower bound, 2 the upper. ``task_weights`` zeroes the two
carrier columns so they cannot contribute error. Note this inflates Chemprop's reported
loss by a constant factor, since its denominator counts all three columns; gradients
are unaffected.

**Scaling.** Chemprop's own ``normalize_targets`` standardises each label column
separately, which would give the value and its two bounds three different scales and
destroy the relationship between them -- a prediction could land "inside" an interval
that no longer surrounds it. The three columns are therefore standardised together
here, by the value column's own centre and spread, and the transform is inverted at
prediction time.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data as cpdata
from chemprop import models as cpmodels
from chemprop import nn as cpnn
from chemprop.nn.metrics import ChempropMetric
from lightning import pytorch as pl
from numpy.typing import NDArray
from torch import nn

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import (
    ASSAYS,
    SOURCES,
    encode_conditions,
    n_condition_columns,
)
from cypshift.features.embeddings import embedding_block, embedding_width
from cypshift.features.ionization import block_for as ionization_block_for
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.shared import SCORED_ASSAY
from cypshift.models.tdi import TDI_FLOOR_DEFAULT, TDI_SHIFT_DEFAULT
from cypshift.structure.cofold import STRUCTURE_FEATURE_NAMES
from cypshift.structure.features import structure_block

logger = logging.getLogger(__name__)

#: Label columns: the value, then the two carrier columns holding its bounds, then one column
#: per response task (a qHTS max-response and/or the screen fraction) from RESPONSE onward.
#: Per-row routing is by NaN: a pIC50 row leaves every response column NaN, a response row leaves
#: VALUE/LOWER/UPPER and the other response columns NaN, and chemprop's finiteness mask zeroes the
#: irrelevant tasks, so the scales never mix. RESPONSE is the index of the first response column.
VALUE, LOWER, UPPER, RESPONSE = 0, 1, 2, 3
N_LABEL_COLUMNS = 3

#: Objectives whose value-task loss is a two-sided point penalty. A censored (one-sided) row
#: has no single value for them to train toward -- its label is "at most this" -- so its point
#: target is masked out under these, unlike the soft-threshold loss, whose dead-zone reads the
#: one-sided bound from the carriers and so keeps the row.
POINT_OBJECTIVES = ("absolute", "squared", "expectile", "pinball")

#: A censored row carries ``lower = -1e4`` before standardisation (see ``_datapoints``), so
#: after standardisation its LOWER carrier sits thousands of units below any real bound. The
#: point and ranking augmentations use this to tell a one-sided censored reading from a real
#: (two-sided) credible interval, in standardised units where a real bound is within a few.
_CENSORED_LOWER_SENTINEL = -100.0
#: The preincubated dose-response assay_condition's assay tag (see cypshift.data.assembly.ASSAYS),
#: the
#: second assay_condition predict_is_time_dependent compares against the direct one.
TDI_ASSAY = "tdi_drc"

#: CheMeleon: a D-MPNN message-passing encoder pretrained on 100M molecules, published on
#: Zenodo. Loaded into chemprop's own BondMessagePassing and fine-tuned here, the same way
#: chemprop's CLI consumes it. Cite DOI 10.48550/arXiv.2506.15792 when used.
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def chemeleon_message_passing(cache_path: Path) -> cpnn.BondMessagePassing:
    """The pretrained CheMeleon encoder, downloaded to ``cache_path`` on first use.

    Our default atom/bond featuriser already matches CheMeleon's (d_v=72, d_e=14), so the
    encoder drops in without a featuriser change; its weights initialise training and are
    fine-tuned, not frozen.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        import requests

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading CheMeleon encoder to %s", cache_path)
        with requests.get(CHEMELEON_URL, stream=True, timeout=180) as response:
            response.raise_for_status()
            with cache_path.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
    checkpoint = torch.load(cache_path, weights_only=True)
    block = cpnn.BondMessagePassing(**checkpoint["hyper_parameters"])
    block.load_state_dict(checkpoint["state_dict"])
    logger.info("loaded CheMeleon encoder (output dim %d)", block.output_dim)
    return block


class SoftThresholdLoss(ChempropMetric):
    """Soft-threshold absolute error on the value task, reading its bounds from carriers.

    With a fourth (RESPONSE) task present, it also returns a squared error there -- the
    qHTS max-response or screen-response auxiliary head. Each row carries only one of the
    two (the other is NaN upstream, so chemprop's finiteness mask zeroes it), so the losses
    never mix.

    ``point_weight`` adds ``point_weight * (pred - midpoint)**2`` to the value-task loss on
    uncensored rows (midpoint = (lower + upper) / 2); ``0.0`` disables it. Censored rows (lower
    at the ``_CENSORED_LOWER_SENTINEL`` floor) are excluded from the point term.
    """

    def __init__(self, point_weight: float = 0.0, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.point_weight = point_weight

    def _calc_unreduced_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        prediction = preds[:, VALUE]
        above = torch.clamp(prediction - targets[:, UPPER], min=0.0)
        below = torch.clamp(targets[:, LOWER] - prediction, min=0.0)
        loss = torch.zeros_like(targets)
        loss[:, VALUE] = above + below
        if self.point_weight > 0.0:
            reliable = targets[:, LOWER] > _CENSORED_LOWER_SENTINEL
            midpoint = 0.5 * (targets[:, LOWER] + targets[:, UPPER])
            loss[:, VALUE] = (
                loss[:, VALUE] + self.point_weight * reliable * (prediction - midpoint) ** 2
            )
        if targets.shape[1] > RESPONSE:
            # Every response task (columns RESPONSE onward) keeps a plain squared error; chemprop's
            # finiteness mask zeroes the columns a given row does not carry.
            loss[:, RESPONSE:] = (preds[:, RESPONSE:] - targets[:, RESPONSE:]) ** 2
        return loss


class RankingSoftThresholdLoss(SoftThresholdLoss):
    """Soft-threshold loss plus an interval-aware pairwise ranking hinge within each batch.

    Over the value-task rows of a batch whose credible intervals are disjoint (the pairs with
    a defined order), adds ``max(0, margin - gap)`` where ``gap`` is the predicted difference
    in the intervals' order, weighted by the weaker row's observation weight. Scaled by
    ``ranking_weight`` (``0.0`` recovers the plain loss).

    The term is added in :meth:`update`, not through the per-row loss tensor, because a
    pairwise penalty is not a per-row quantity. It is normalised to a weighted per-pair mean
    and multiplied by the eligible-row count, matching the scale of the summed per-row loss
    that ``compute()`` divides by ``num_samples``, so ``ranking_weight`` is batch-size
    independent.
    """

    def __init__(
        self, ranking_weight: float = 0.0, ranking_margin: float = 0.5, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        *args: torch.Tensor,
    ) -> None:
        super().update(preds, targets, mask, weights, *args)
        if self.ranking_weight <= 0.0:
            return
        # chemprop passes the finiteness mask separately and has already replaced NaN targets
        # with zeros, so the value rows are mask[:, VALUE], not targets.isfinite(). Only rows
        # with a real two-sided interval (not a censored one-sided floor) can be ordered, and
        # the ranking term honours the observation weights the rest of the loss does: a
        # zero-weight row (a dropped source, a balanced-away endpoint) takes no part, and each
        # pair counts by its weaker member's weight. Pairs are every disjoint-interval pair in
        # the batch regardless of target/assay; restricting them to same-endpoint neighbours is
        # a separate modelling choice the loss has no row identity to make here.
        if mask is None:
            return
        row_weight = (
            torch.ones(targets.shape[0], device=targets.device)
            if weights is None
            else weights.reshape(-1).to(targets.device)
        )
        value_mask = (
            mask[:, VALUE] & (targets[:, LOWER] > _CENSORED_LOWER_SENTINEL) & (row_weight > 0.0)
        )
        if int(value_mask.sum()) < 2:
            return
        pred = preds[value_mask, VALUE]
        lower = targets[value_mask, LOWER]
        upper = targets[value_mask, UPPER]
        weight = row_weight[value_mask]
        # Pair (i, j) is ordered i<j exactly where upper_i < lower_j (the intervals are
        # disjoint and i sits below j), and its weight is the weaker row's. Penalise when the
        # predicted gap fails to clear the margin in that direction.
        ordered = (upper.unsqueeze(1) < lower.unsqueeze(0)).float()  # (i, j): i strictly below j
        n_ordered = ordered.sum()
        if float(n_ordered) == 0.0:
            return
        pair_weight = torch.minimum(weight.unsqueeze(1), weight.unsqueeze(0)) * ordered
        gap = pred.unsqueeze(0) - pred.unsqueeze(1)  # (i, j): pred_j - pred_i
        hinge = torch.clamp(self.ranking_margin - gap, min=0.0)
        # Normalise by the unweighted ordered-pair count, not the summed pair weight. Dividing
        # by the pair weight would cancel any common weight scale, so uniformly down-weighting a
        # source (e.g. all its rows x0.1) would leave its ranking pull untouched while its
        # interval loss shrank tenfold. Dividing by the count keeps the term proportional to the
        # absolute pair weights, the same budget the per-row interval loss honours. (For unit
        # weights this equals the old per-pair mean.) value_mask.sum() puts it on the scale of
        # the summed per-row loss compute() divides by num_samples, so ranking_weight stays a
        # relative weight independent of batch size. Accumulate in place to keep the state buffer.
        penalty = (hinge * pair_weight).sum() / n_ordered
        self.total_loss += self.ranking_weight * penalty * value_mask.sum()


class _AsymmetricValueLoss(ChempropMetric):
    """Asymmetric point loss on the value task; ``tau>0.5`` penalises under-prediction more.

    Base for expectile (asymmetric squared) and pinball (asymmetric absolute). Both keep a
    gradient everywhere, so they preserve the predicted spread rather than collapsing it the
    way the dead-zone soft-threshold loss does. The response auxiliary task, when present,
    keeps its squared error. Censored rows carry no point value (NaN), so chemprop's
    finiteness mask drops them here.
    """

    def __init__(self, tau: float = 0.5, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.tau = tau

    def _per_row(self, prediction: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _calc_unreduced_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros_like(targets)
        loss[:, VALUE] = self._per_row(preds[:, VALUE], targets[:, VALUE])
        if targets.shape[1] > RESPONSE:
            # Every response task keeps a plain squared error (masked per row by chemprop).
            loss[:, RESPONSE:] = (preds[:, RESPONSE:] - targets[:, RESPONSE:]) ** 2
        return loss


class ExpectileLoss(_AsymmetricValueLoss):
    """Asymmetric least squares: ``w*(y-f)^2`` with ``w=tau`` when under-predicting."""

    def _per_row(self, prediction: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        side = torch.where(value > prediction, self.tau, 1.0 - self.tau)
        return side * (value - prediction) ** 2


class PinballLoss(_AsymmetricValueLoss):
    """Quantile loss ``max(tau*u, (tau-1)*u)``, ``u = y - f``, eliciting the tau quantile."""

    def _per_row(self, prediction: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        residual = value - prediction
        return torch.maximum(self.tau * residual, (self.tau - 1.0) * residual)


class _TargetAdaptedRegressionFFN(cpnn.RegressionFFN):
    """RegressionFFN with a gated, isoform-specific residual on the value task.

    An MLP over the predictor input ``Z`` (the pooled graph representation concatenated with
    the conditioning vector) outputs a scalar added to the shared value prediction, multiplied
    by the gate ``Z[:, target_offset + target_index]`` -- that isoform's target one-hot, which
    ``fingerprint`` places at a fixed offset after the encoder output (the target block is the
    first conditioning block). The gate is 0 for every other isoform and the auxiliary enzymes,
    so their output is unchanged; the encoder and the rest of the readout are shared. The
    adapter's last layer is zero-initialised, so it starts as a no-op.
    """

    def __init__(
        self,
        *,
        adapt_hidden: int,
        target_offset: int,
        target_index: int,
        value_index: int = VALUE,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.target_offset = target_offset
        self.target_index = target_index
        self.value_index = value_index
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, adapt_hidden),
            nn.ReLU(),
            nn.Linear(adapt_hidden, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # noqa: N803 -- chemprop's Predictor API
        base = self.ffn(Z)
        gate = Z[:, self.target_offset + self.target_index].unsqueeze(1)
        correction = self.adapter(Z) * gate  # (batch, 1); zero wherever the gate is zero
        padded = torch.zeros_like(base)
        padded[:, self.value_index : self.value_index + 1] = correction
        return self.output_transform(base + padded)

    train_step = forward


class _PerTargetHeadsRegressionFFN(cpnn.RegressionFFN):
    """RegressionFFN with an independent value-task output head per target on the shared trunk.

    The shared ``ffn`` still produces the carrier and response columns, but the value task is
    read from a per-target head selected by that row's target one-hot: each target gets its own
    small FFN on the shared encoder+conditioning ``Z``, so the readout is target-specific while
    the message-passing encoder stays shared. Distinct from ``separate_trunks`` (a separate
    encoder per isoform) and the adapter (one shared value head plus a gated residual); this is
    the shared-trunk, independent-output-head arrangement. The target one-hot is the first
    conditioning block, so it sits at ``target_offset : target_offset + n_targets`` in ``Z``.

    ``head_hidden`` is the per-target head's hidden width and ``head_dropout`` its dropout, both
    wired from the graph config (``ffn_hidden`` and the model dropout) so this readout's capacity
    and regularisation match the shared FFN rather than being fixed -- a ``per_target_heads +
    ffn_hidden`` sweep then actually varies the potency readout's width, and the head carries the
    same dropout as the shared readout so the comparison is output sharing, not regularisation.
    """

    def __init__(
        self,
        *,
        target_offset: int,
        n_targets: int,
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        value_index: int = VALUE,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.target_offset = target_offset
        self.n_targets = n_targets
        self.value_index = value_index
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, head_hidden),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
            for _ in range(n_targets)
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # noqa: N803 -- chemprop's Predictor API
        base = self.ffn(Z)
        gates = Z[:, self.target_offset : self.target_offset + self.n_targets]  # (batch, targets)
        per_target = torch.cat([head(Z) for head in self.heads], dim=1)  # (batch, targets)
        # Each row's target one-hot selects its own head's value; the carrier/response columns
        # keep the shared ffn's output.
        value = (gates * per_target).sum(dim=1)
        replaced = base.clone()
        replaced[:, self.value_index] = value
        return self.output_transform(replaced)

    train_step = forward


class _BatchedHeads(nn.Module):
    """``n_heads`` independent two-layer MLP heads on one input, evaluated in a single einsum.

    A ``ModuleList`` of small heads costs a Python loop and a handful of kernel launches per head
    per step, which is slow when the number of heads is a product of two vocabularies. Stacking
    the weights as ``(heads, ...)`` tensors evaluates every head at once. Each head is a separate
    set of weights and shares nothing with the others; initialisation matches ``nn.Linear``.
    """

    def __init__(self, n_heads: int, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        bound_in, bound_out = 1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(hidden)
        self.w1 = nn.Parameter(
            torch.empty(n_heads, input_dim, hidden).uniform_(-bound_in, bound_in)
        )
        self.b1 = nn.Parameter(torch.empty(n_heads, hidden).uniform_(-bound_in, bound_in))
        self.w2 = nn.Parameter(torch.empty(n_heads, hidden).uniform_(-bound_out, bound_out))
        self.b2 = nn.Parameter(torch.empty(n_heads).uniform_(-bound_out, bound_out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(batch, input_dim) -> (batch, n_heads)``: every head's scalar output."""
        hidden = torch.relu(torch.einsum("bi,pih->bph", z, self.w1) + self.b1)
        return torch.einsum("bph,ph->bp", self.dropout(hidden), self.w2) + self.b2


class _MeasurementTableRegressionFFN(cpnn.RegressionFFN):
    """RegressionFFN reading each ``(target, measurement)`` pair from its own output head.

    The potency (value) task has one independent head per ``(target, assay)`` pair, selected by
    the outer product of the row's target and assay one-hots; every response task (a qHTS max
    response, a screen fraction, a raw log2FC) has one independent head per target, selected by
    the target one-hot. Pairs a row never carries are masked by the loss and never trained, so
    the table is the "masked multi-task output table" of the reference recipes, with the message-
    passing encoder shared. A head sees only its own measurements: unlike a conditioned shared
    readout, an auxiliary assay (Octant, Tox21, the screen) cannot move the scored assay's output
    directly and informs it only through the shared encoder.

    Distinct from ``per_target_heads`` (one potency head per target, shared across assays) and
    from ``separate_trunks`` (a separate encoder per isoform). The target and assay one-hots are
    the first two conditioning blocks, so they sit at ``target_offset`` and ``assay_offset`` in
    ``Z``; that layout only exists for ``factorised`` conditioning.
    """

    def __init__(
        self,
        *,
        target_offset: int,
        n_targets: int,
        assay_offset: int,
        n_assays: int,
        n_response: int,
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        value_index: int = VALUE,
        response_index: int = RESPONSE,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.target_offset, self.n_targets = target_offset, n_targets
        self.assay_offset, self.n_assays = assay_offset, n_assays
        self.value_index, self.response_index = value_index, response_index
        self.value_heads = _BatchedHeads(
            n_targets * n_assays, self.input_dim, head_hidden, head_dropout
        )
        self.response_heads = nn.ModuleList(
            _BatchedHeads(n_targets, self.input_dim, head_hidden, head_dropout)
            for _ in range(n_response)
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # noqa: N803 -- chemprop's Predictor API
        base = self.ffn(Z)
        targets = Z[:, self.target_offset : self.target_offset + self.n_targets]
        assays = Z[:, self.assay_offset : self.assay_offset + self.n_assays]
        # The (target, assay) one-hot pair is a one-hot over the product vocabulary.
        pair = (targets.unsqueeze(2) * assays.unsqueeze(1)).reshape(len(Z), -1)
        replaced = base.clone()
        replaced[:, self.value_index] = (pair * self.value_heads(Z)).sum(dim=1)
        for task, heads in enumerate(self.response_heads):
            replaced[:, self.response_index + task] = (targets * heads(Z)).sum(dim=1)
        return self.output_transform(replaced)

    train_step = forward


class GraphNetwork:
    """Chemprop D-MPNN conditioned on target and assay through its readout."""

    name = "graph_network"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        assays: tuple[str, ...] = ASSAYS,
        conditioning: str = "factorised",
        hidden: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 64,
        num_workers: int = 0,
        n_members: int = 1,
        loss: str = "soft_threshold",
        tau: float = 0.5,
        point_loss_weight: float = 0.0,
        ranking_loss_weight: float = 0.0,
        ranking_margin: float = 0.5,
        ensemble: str = "reseed",
        pretraining: str = "none",
        pretraining_cache: str = "data/external/chemeleon/chemeleon_mp.pt",
        pretraining_finetune: bool = True,
        source_heads: bool = False,
        adapt_target: str = "none",
        adapt_hidden: int = 64,
        per_target_heads: bool = False,
        per_measurement_heads: bool = False,
        ffn_hidden: int = 0,
        learning_rate: float = 0.0,
        warmup_epochs: int = 2,
        response_assays: tuple[str, ...] = (),
        tdi_shift_threshold: float = TDI_SHIFT_DEFAULT,
        tdi_assay_floor: float = TDI_FLOOR_DEFAULT,
        physchem: bool = False,
        mechanism: bool = False,
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
        use_structure: bool = False,
        structure_cache: str = "data/structure",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        # Each response task is trained on its own assay (a qHTS max-response and/or a screen
        # fraction); every such assay joins the conditioning vocabulary and the label matrix
        # grows one column per response task (from the RESPONSE index onward). They coexist,
        # each on its own column and its own 0-1 scale, so the panel can carry several
        # measurements at once rather than one.
        self.response_assays = tuple(response_assays)
        self.auxiliary_response = bool(self.response_assays)
        self.tdi_shift_threshold = tdi_shift_threshold
        self.tdi_assay_floor = tdi_assay_floor
        self.assays = (*assays, *self.response_assays)
        self._n_label = N_LABEL_COLUMNS + len(self.response_assays)
        self.conditioning = conditioning
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        # Dataloader worker processes. 0 loads in the main process (safe everywhere, but
        # starves the GPU); more parallelise featurisation so the GPU stays fed. Kept
        # settable because the right value depends on the machine and who else is using it.
        self.num_workers = num_workers
        self.n_members = n_members
        self.loss = loss
        self.tau = tau
        #: Extra terms layered onto the soft-threshold loss (ignored by the other objectives):
        #: a squared pull to the midpoint on reliable rows, and an interval-aware pairwise
        #: ranking hinge. Both default off.
        self.point_loss_weight = point_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_margin = ranking_margin
        #: One scored isoform to give a gated, isoform-specific adaptation block, or "none".
        self.adapt_target = adapt_target
        self.adapt_hidden = adapt_hidden
        #: Give every target its own value-task output head on the shared encoder (shared trunk,
        #: independent heads), rather than one shared readout. Mutually exclusive with the adapter.
        self.per_target_heads = per_target_heads
        #: Give every ``(target, measurement)`` pair its own output head on the shared encoder (see
        #: :class:`_MeasurementTableRegressionFFN`). Needs the separable target/assay one-hots of
        #: ``factorised`` conditioning; mutually exclusive with the other readout variants.
        self.per_measurement_heads = per_measurement_heads
        if per_measurement_heads and conditioning != "factorised":
            raise ValueError(
                "per_measurement_heads reads the target and assay one-hots, which only "
                f"conditioning='factorised' lays out; got {conditioning!r}"
            )
        #: Readout FFN width, independent of the message-passing width; 0 keeps chemprop's default.
        self.ffn_hidden = ffn_hidden
        #: Peak rate of chemprop's warmup / decay schedule (0 keeps its 1e-4 / 1e-3 / 1e-4) and the
        #: warmup length; the initial and final rates follow a set peak at a tenth of it.
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        #: How ensemble members differ ("reseed" or "bagging") and which pretrained encoder
        #: to initialise from ("none"/"chemeleon"/"mammal").
        self.ensemble = ensemble
        self.pretraining = pretraining
        self.pretraining_cache = pretraining_cache
        #: Fine-tune the pretrained encoder (default) or freeze it as a fixed featuriser so
        #: only the readout trains. Only meaningful when pretraining loads an encoder.
        self.pretraining_finetune = pretraining_finetune
        #: Condition on which dataset each row came from (a source one-hot), so same-chemistry
        #: sources kept apart in assembly each get their own calibration. Swept.
        self.source_heads = source_heads
        #: Append the Boltz-2 cofold structure block (affinity + heme-coordination pose) to the
        #: conditioning. Needs sources.structure and a populated cache; NaN (uncofolded) is
        #: imputed to zero since the network cannot take NaN. Swept.
        self.use_structure = use_structure
        self.structure_cache = structure_cache
        self.seed = seed
        self._members: list[cpmodels.MPNN] = []
        self._conditioning_sources = SOURCES if source_heads else None
        self._n_conditioning = n_condition_columns(
            conditioning, targets, self.assays, sources=self._conditioning_sources
        )
        if use_structure:
            self._n_conditioning += len(STRUCTURE_FEATURE_NAMES)
        # "mammal" pretraining appends the cached MAMMAL embedding to the readout's
        # conditioning slot (x_d) as fixed input features; widen the readout to fit it.
        self._embedding = embedding_block("mammal") if pretraining == "mammal" else None
        if self._embedding is not None:
            self._n_conditioning += embedding_width("mammal")
        # Computed descriptor blocks (physchem / mechanistic / ionization geometry) the
        # learned graph cannot derive, appended to the readout's conditioning slot. The
        # graph's own fingerprint is left off -- the message passing replaces it.
        self._descriptor = None
        if physchem or mechanism or ionization_geometry:
            ionization = ionization_block_for(ionization_geometry, Path(ionization_cache))

            def descriptor_block(smiles: list[str]) -> np.ndarray:
                return featurise(
                    smiles,
                    fingerprint=False,
                    physchem=physchem,
                    mechanism=mechanism,
                    ionization=ionization,
                )

            self._descriptor = descriptor_block
            self._n_conditioning += descriptor_block(["CCO"]).shape[1]
        self._centre = 0.0
        self._spread = 1.0
        #: Per-response-task standardisation (one entry per self.response_assays), so each task's
        #: own 0-1 scale is not forced onto the pIC50 centre/spread or onto another task's.
        self._resp_centre = np.zeros(len(self.response_assays))
        self._resp_spread = np.ones(len(self.response_assays))
        #: Standardisation of the appended continuous x_d blocks (embedding, descriptors),
        #: fitted on retained training; the one-hot conditioning is left raw.
        self._xd_centre: np.ndarray | None = None
        self._xd_scale: np.ndarray | None = None

    def _continuous_blocks(self, smiles: list[str]) -> np.ndarray:
        """The appended continuous x_d blocks (embedding, computed descriptors), or empty.

        Both blocks are functions of the structure alone, so each distinct SMILES is featurised once
        and its row is repeated for every observation of that compound.
        """
        if self._embedding is None and self._descriptor is None:
            return np.empty((len(smiles), 0), dtype=np.float32)
        codes, distinct = pd.factorize(pd.Series(smiles, dtype=object))
        distinct = distinct.tolist()
        blocks = []
        if self._embedding is not None:
            blocks.append(self._embedding(distinct))
        if self._descriptor is not None:
            blocks.append(self._descriptor(distinct))
        return np.hstack(blocks)[codes]

    def _datapoints(self, frame: pd.DataFrame, labelled: bool) -> list[cpdata.MoleculeDatapoint]:
        conditioning = encode_conditions(
            frame, self.conditioning, self.targets, self.assays, sources=self._conditioning_sources
        )
        continuous = self._continuous_blocks(frame["SMILES"].tolist())
        if continuous.shape[1] and self._xd_centre is not None:
            # Standardise the appended continuous blocks (embedding, descriptors) fitted on
            # retained training; the sparse one-hot conditioning is left raw.
            continuous = (continuous - self._xd_centre) / self._xd_scale
        if continuous.shape[1]:
            conditioning = np.hstack([conditioning, continuous])
        if self.use_structure:
            # Cofold block is per-(target, SMILES); NaN (uncofolded) imputed to zero because
            # the network, unlike a tree, cannot take NaN. Appended raw (not standardised).
            structure = np.nan_to_num(structure_block(frame, Path(self.structure_cache)), nan=0.0)
            conditioning = np.hstack([conditioning, structure]).astype(np.float32)
        smiles = frame["SMILES"].tolist()
        # Each distinct SMILES is parsed once; the datapoints of one compound share the molecule.
        molecules = {
            smi: cpdata.MoleculeDatapoint.from_smi(smi).mol for smi in dict.fromkeys(smiles)
        }
        if labelled:
            assays = frame["assay"].tolist()
            values = frame["value"].to_numpy(float)
            lowers = frame["conf_low"].to_numpy(float)
            uppers = frame["conf_high"].to_numpy(float)
            censored_rows = (
                frame["is_censored"].to_numpy(bool)
                if "is_censored" in frame.columns
                else np.zeros(len(frame), dtype=bool)
            )
            weights = frame["weight"].to_numpy(float)
        points = []
        for row, smi in enumerate(smiles):
            if labelled:
                y = np.full(self._n_label, np.nan)
                if self.auxiliary_response and assays[row] in self.response_assays:
                    # A response row: fill only its own response column, leave the pIC50 tasks and
                    # the other response columns NaN, each on its own standardisation.
                    task = self.response_assays.index(assays[row])
                    y[RESPONSE + task] = (
                        values[row] - self._resp_centre[task]
                    ) / self._resp_spread[task]
                else:
                    value = values[row]
                    lower = lowers[row] if np.isfinite(lowers[row]) else value
                    upper = uppers[row] if np.isfinite(uppers[row]) else value
                    censored = bool(censored_rows[row])
                    if censored:
                        lower = -1e4
                    y[VALUE : UPPER + 1] = (
                        np.array([value, lower, upper], dtype=float) - self._centre
                    ) / self._spread
                    if censored and self.loss in POINT_OBJECTIVES:
                        # A two-sided point objective has no value to train a censored row
                        # toward; NaN masks its value task so it is dropped rather than pulled
                        # to the clip. The soft-threshold loss keeps it (its dead-zone reads the
                        # one-sided bound from the carriers).
                        y[VALUE] = np.nan
                weight = float(weights[row])
            else:
                y = np.zeros(self._n_label)
                weight = 1.0
            points.append(
                cpdata.MoleculeDatapoint(
                    molecules[smi],
                    y=y,
                    x_d=conditioning[row].astype(float),
                    weight=weight,
                    name=smi,
                )
            )
        return points

    def _build(self) -> cpmodels.MPNN:
        if self.pretraining == "chemeleon":
            message_passing = chemeleon_message_passing(Path(self.pretraining_cache))
            if not self.pretraining_finetune:
                # Freeze the pretrained encoder: it stays a fixed featuriser and only the
                # readout trains. Cheaper and overfit-resistant on small data, at the cost
                # of not adapting the representation to CYP chemistry.
                for parameter in message_passing.parameters():
                    parameter.requires_grad = False
        else:
            message_passing = cpnn.BondMessagePassing(
                d_h=self.hidden, depth=self.depth, dropout=self.dropout
            )
        # Value task on, the lower/upper carriers off, the response task (when present) on.
        # Every objective installs an explicit criterion with these weights: leaving it None
        # gave chemprop's default MSE over all tasks, which trained the carrier columns and
        # made "absolute" an MSE run rather than MAE.
        weights = torch.tensor([[1.0, 0.0, 0.0] + [1.0] * len(self.response_assays)])
        if self.loss == "soft_threshold":
            # The point and ranking augmentations layer onto the dead-zone loss; the ranking
            # hinge needs the batch, so it is a distinct criterion when it is switched on.
            if self.ranking_loss_weight > 0.0:
                criterion = RankingSoftThresholdLoss(
                    ranking_weight=self.ranking_loss_weight,
                    ranking_margin=self.ranking_margin,
                    point_weight=self.point_loss_weight,
                    task_weights=weights,
                )
            else:
                criterion = SoftThresholdLoss(
                    point_weight=self.point_loss_weight, task_weights=weights
                )
        elif self.loss == "absolute":
            criterion = cpnn.MAE(task_weights=weights)
        elif self.loss == "squared":
            criterion = cpnn.MSE(task_weights=weights)
        elif self.loss == "expectile":
            criterion = ExpectileLoss(self.tau, task_weights=weights)
        elif self.loss == "pinball":
            criterion = PinballLoss(self.tau, task_weights=weights)
        else:
            raise ValueError(f"unknown objective {self.loss!r}")
        input_dim = message_passing.output_dim + self._n_conditioning
        # Readout FFN width, set independently of the encoder width; 0 keeps chemprop's default.
        ffn_kwargs = {} if self.ffn_hidden <= 0 else {"hidden_dim": self.ffn_hidden}
        if self.adapt_target != "none":
            # The target one-hot is the first block of the conditioning, which fingerprint()
            # places right after the encoder output, so its gate column sits at
            # message_passing.output_dim + (index of the adapted isoform in self.targets).
            predictor = _TargetAdaptedRegressionFFN(
                adapt_hidden=self.adapt_hidden,
                target_offset=message_passing.output_dim,
                target_index=self.targets.index(self.adapt_target),
                n_tasks=self._n_label,
                input_dim=input_dim,
                criterion=criterion,
                dropout=self.dropout,
                **ffn_kwargs,
            )
        elif self.per_target_heads:
            # Shared encoder, one value-task output head per target, gated by the target one-hot
            # (the first conditioning block, placed right after the encoder output). The head
            # width follows ffn_hidden (so a per_target_heads + ffn_hidden sweep varies the
            # potency readout, not just the shared carrier/response FFN) and it carries the model
            # dropout, so it is not a fixed, unregularised 128-wide block.
            predictor = _PerTargetHeadsRegressionFFN(
                target_offset=message_passing.output_dim,
                n_targets=len(self.targets),
                head_hidden=self.ffn_hidden if self.ffn_hidden > 0 else 128,
                head_dropout=self.dropout,
                n_tasks=self._n_label,
                input_dim=input_dim,
                criterion=criterion,
                dropout=self.dropout,
                **ffn_kwargs,
            )
        elif self.per_measurement_heads:
            # Shared encoder, one output head per (target, assay) for potency and one per target
            # for each response task. The one-hot blocks are the first two conditioning blocks,
            # placed right after the encoder output. Head width and dropout follow the same rule as
            # the per-target heads (ffn_hidden when set, else 128; the model dropout).
            predictor = _MeasurementTableRegressionFFN(
                target_offset=message_passing.output_dim,
                n_targets=len(self.targets),
                assay_offset=message_passing.output_dim + len(self.targets),
                n_assays=len(self.assays),
                n_response=len(self.response_assays),
                head_hidden=self.ffn_hidden if self.ffn_hidden > 0 else 128,
                head_dropout=self.dropout,
                n_tasks=self._n_label,
                input_dim=input_dim,
                criterion=criterion,
                dropout=self.dropout,
                **ffn_kwargs,
            )
        else:
            predictor = cpnn.RegressionFFN(
                n_tasks=self._n_label,
                input_dim=input_dim,
                criterion=criterion,
                dropout=self.dropout,
                **ffn_kwargs,
            )
        schedule: dict[str, float] = {"warmup_epochs": self.warmup_epochs}
        if self.learning_rate > 0:
            schedule.update(
                init_lr=self.learning_rate / 10,
                max_lr=self.learning_rate,
                final_lr=self.learning_rate / 10,
            )
        return cpmodels.MPNN(message_passing, cpnn.MeanAggregation(), predictor, **schedule)

    @staticmethod
    def _log_capacity(model: cpmodels.MPNN) -> None:
        """Log the built network's size: encoder and readout parameters, and the encoder's output
        width. Equal ``hidden`` / ``ffn_hidden`` settings do not mean equal capacity (independent
        heads replicate an MLP per task, and a pretrained checkpoint fixes the encoder width), so a
        readout comparison is read against these counts."""

        def count(module: object) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        logger.info(
            "graph network: %d parameters (encoder %d, readout %d); encoder output width %d",
            count(model),
            count(model.message_passing),
            count(model.predictor),
            model.message_passing.output_dim,
        )

    def fit(self, observations: pd.DataFrame) -> "GraphNetwork":
        """Train on the observation table."""
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")
        # The pIC50 rows set one centre/spread for the value and its two bound columns; each
        # response task's rows (a different 0-1 scale) are standardised separately, per assay.
        is_response = usable["assay"].isin(self.response_assays).to_numpy()
        pic50 = usable.loc[~is_response, "value"].to_numpy(float)
        self._centre = float(pic50.mean())
        self._spread = float(pic50.std()) or 1.0
        for task, assay in enumerate(self.response_assays):
            rows = usable.loc[usable["assay"] == assay, "value"].to_numpy(float)
            if rows.size:
                self._resp_centre[task] = float(rows.mean())
                self._resp_spread[task] = float(rows.std()) or 1.0
        # Fit the continuous-x_d standardisation on the retained training rows, before the
        # datapoints apply it.
        continuous = self._continuous_blocks(usable["SMILES"].tolist())
        if continuous.shape[1]:
            self._xd_centre = continuous.mean(axis=0)
            self._xd_scale = continuous.std(axis=0) + 1e-6
        logger.info(
            "building %d datapoints (%d response), pIC50 centre %.2f spread %.2f",
            len(usable),
            int(is_response.sum()),
            self._centre,
            self._spread,
        )
        points = self._datapoints(usable, labelled=True)

        self._members = []
        for member in range(self.n_members):
            pl.seed_everything(self.seed + member, workers=True)
            if self.ensemble == "bagging":
                # Each member sees a bootstrap resample, so members differ in the data
                # they fit, not only in initialisation -- the diversity a reseed lacks.
                rng = np.random.default_rng(self.seed + member)
                member_points = [points[i] for i in rng.integers(0, len(points), size=len(points))]
            else:
                member_points = points
            dataset = cpdata.MoleculeDataset(member_points)
            # Featurise every molecule graph once, before training, so an epoch only collates
            # cached graphs instead of re-featurising each molecule (forked loader workers read the
            # cache without copying it).
            dataset.cache = True
            loader = cpdata.build_dataloader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            model = self._build()
            if member == 0:
                self._log_capacity(model)
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator="auto",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(model, loader)
            self._members.append(model)
            logger.info("member %d trained", member)
        return self

    def _samples_at_assay(self, smiles: list[str], assay: str) -> NDArray[np.float64]:
        """Per-member predicted pIC50 at one assay condition; (members, n, endpoints)."""
        if not self._members:
            raise RuntimeError("model is not fitted")
        n = len(smiles)
        samples = np.empty((self.n_members, n, len(self.endpoints)))
        for column, endpoint in enumerate(self.endpoints):
            frame = pd.DataFrame(
                {
                    "SMILES": smiles,
                    "target": [endpoint.split("_")[0]] * n,
                    "assay": [assay] * n,
                    # Predict as the reference source; source-stratified heads read this.
                    "source": ["challenge"] * n,
                }
            )
            dataset = cpdata.MoleculeDataset(self._datapoints(frame, labelled=False))
            loader = cpdata.build_dataloader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            for index, model in enumerate(self._members):
                model.eval()
                with torch.no_grad():
                    predicted = torch.cat(
                        [model(batch.bmg, batch.V_d, batch.X_d) for batch in loader]
                    )
                samples[index, :, column] = (
                    predicted[:, VALUE].numpy() * self._spread + self._centre
                )
        return samples

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict at the scored (direct-inhibition) assay condition for every endpoint."""
        return PredictiveDistribution(self._samples_at_assay(smiles, SCORED_ASSAY), self.endpoints)

    def predict_is_time_dependent(self, smiles: list[str]) -> NDArray[np.bool_]:
        """Call is_TDI by comparing the direct and preincubated assay_conditions the graph predicts.


        Same two-views-of-one-object idea as the curve model: condition the one trunk on the
        direct and the ``tdi_drc`` (preincubated) assay, then apply the organisers' shift rule
        with the model's own swept thresholds. Only meaningful when the preincubated assay_condition
        was
        trained (``sources.tdi_assay_condition``); run_cv scores TDI only in that case.
        """
        direct = self._samples_at_assay(smiles, SCORED_ASSAY).mean(axis=0)
        preincubated = self._samples_at_assay(smiles, TDI_ASSAY).mean(axis=0)
        floor, shift = self.tdi_assay_floor, self.tdi_shift_threshold
        return np.where(
            direct > floor,
            (preincubated - direct) > shift,
            preincubated > floor + shift,
        )


class SeparatedTrunkGraph:
    """One GraphNetwork per scored endpoint: a separate encoder per isoform, no trunk shared
    across the scored isoforms.

    The shared-trunk GraphNetwork forces one representation to serve all four isoforms at
    once. That interferes with CYP2D6, whose potency tracks a protonated basic amine and does
    not ride the lipophilicity axis the other three share, so a trunk shaped by the others
    pulls 2D6 the wrong way. Here each scored isoform trains its own encoder. The auxiliary
    enzymes (relatives) stay in every sub-model for transfer; only the other scored isoforms
    are withheld, so each trunk specialises without their interference. It costs one training
    per isoform. The interface matches GraphNetwork, so it is a drop-in the pipeline selects.
    """

    name = "graph_separated_trunk"

    def __init__(
        self,
        endpoints: tuple[str, ...] = REGRESSION_ENDPOINTS,
        targets: tuple[str, ...] = ISOFORMS,
        **graph_kwargs: object,
    ) -> None:
        self.endpoints = tuple(endpoints)
        self.targets = tuple(targets)
        self._graph_kwargs = graph_kwargs
        scored = {endpoint.split("_")[0] for endpoint in self.endpoints}
        self._auxiliary = tuple(target for target in self.targets if target not in scored)
        self._models: dict[str, GraphNetwork] = {}

    def fit(self, observations: pd.DataFrame) -> "SeparatedTrunkGraph":
        for endpoint in self.endpoints:
            enzyme = endpoint.split("_")[0]
            keep = (enzyme, *self._auxiliary)
            rows = observations[observations["target"].isin(keep)].reset_index(drop=True)
            model = GraphNetwork(endpoints=(endpoint,), targets=keep, **self._graph_kwargs)
            model.fit(rows)
            self._models[endpoint] = model
        return self

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        columns = [self._models[e].predict(smiles).samples[:, :, 0] for e in self.endpoints]
        return PredictiveDistribution(np.stack(columns, axis=2), self.endpoints)

    def predict_is_time_dependent(self, smiles: list[str]) -> NDArray[np.bool_]:
        columns = [self._models[e].predict_is_time_dependent(smiles)[:, 0] for e in self.endpoints]
        return np.stack(columns, axis=1)
