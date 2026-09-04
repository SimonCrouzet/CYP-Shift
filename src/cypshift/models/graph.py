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

import numpy as np
import pandas as pd
import torch
from chemprop import data as cpdata
from chemprop import models as cpmodels
from chemprop import nn as cpnn
from chemprop.nn.metrics import ChempropMetric
from lightning import pytorch as pl

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import ASSAYS, encode_conditions, n_condition_columns
from cypshift.models.base import PredictiveDistribution
from cypshift.models.shared import SCORED_ASSAY

logger = logging.getLogger(__name__)

#: Label columns: the value, then the two carrier columns holding its bounds.
VALUE, LOWER, UPPER = 0, 1, 2
N_LABEL_COLUMNS = 3


class SoftThresholdLoss(ChempropMetric):
    """Soft-threshold absolute error, reading its bounds from carrier label columns."""

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
        return loss


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
        n_members: int = 1,
        loss: str = "soft_threshold",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        self.assays = assays
        self.conditioning = conditioning
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_members = n_members
        self.loss = loss
        self.seed = seed
        self._members: list[cpmodels.MPNN] = []
        self._n_conditioning = n_condition_columns(conditioning, targets, assays)
        self._centre = 0.0
        self._spread = 1.0

    def _datapoints(self, frame: pd.DataFrame, labelled: bool) -> list[cpdata.MoleculeDatapoint]:
        conditioning = encode_conditions(frame, self.conditioning, self.targets, self.assays)
        points = []
        for row, (_, record) in enumerate(frame.iterrows()):
            if labelled:
                value = float(record["value"])
                lower = float(record["conf_low"]) if np.isfinite(record["conf_low"]) else value
                upper = float(record["conf_high"]) if np.isfinite(record["conf_high"]) else value
                if bool(record.get("is_censored", False)):
                    lower = -1e4
                y = (np.array([value, lower, upper], dtype=float) - self._centre) / self._spread
                weight = float(record["weight"])
            else:
                y = np.zeros(N_LABEL_COLUMNS)
                weight = 1.0
            points.append(
                cpdata.MoleculeDatapoint.from_smi(
                    record["SMILES"], y=y, x_d=conditioning[row].astype(float), weight=weight
                )
            )
        return points

    def _build(self) -> cpmodels.MPNN:
        message_passing = cpnn.BondMessagePassing(
            d_h=self.hidden, depth=self.depth, dropout=self.dropout
        )
        criterion = (
            SoftThresholdLoss(task_weights=torch.tensor([[1.0, 0.0, 0.0]]))
            if self.loss == "soft_threshold"
            else None
        )
        predictor = cpnn.RegressionFFN(
            n_tasks=N_LABEL_COLUMNS,
            input_dim=message_passing.output_dim + self._n_conditioning,
            criterion=criterion,
            dropout=self.dropout,
        )
        return cpmodels.MPNN(message_passing, cpnn.MeanAggregation(), predictor)

    def fit(self, observations: pd.DataFrame) -> "GraphNetwork":
        """Train on the observation table."""
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")
        # One centre and spread for all three label columns, so a value and its bounds
        # stay on the same scale.
        value = usable["value"].to_numpy(float)
        self._centre = float(value.mean())
        self._spread = float(value.std()) or 1.0
        logger.info(
            "building %d datapoints, standardised by centre %.2f spread %.2f",
            len(usable),
            self._centre,
            self._spread,
        )
        dataset = cpdata.MoleculeDataset(self._datapoints(usable, labelled=True))
        loader = cpdata.build_dataloader(dataset, batch_size=self.batch_size, num_workers=0)

        self._members = []
        for member in range(self.n_members):
            pl.seed_everything(self.seed + member, workers=True)
            model = self._build()
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

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """Predict at the scored assay condition for every endpoint."""
        if not self._members:
            raise RuntimeError("model is not fitted")
        n = len(smiles)
        samples = np.empty((self.n_members, n, len(self.endpoints)))
        for column, endpoint in enumerate(self.endpoints):
            frame = pd.DataFrame(
                {
                    "SMILES": smiles,
                    "target": [endpoint.split("_")[0]] * n,
                    "assay": [SCORED_ASSAY] * n,
                }
            )
            dataset = cpdata.MoleculeDataset(self._datapoints(frame, labelled=False))
            loader = cpdata.build_dataloader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
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
        return PredictiveDistribution(samples, self.endpoints)
