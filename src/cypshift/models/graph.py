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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop import data as cpdata
from chemprop import models as cpmodels
from chemprop import nn as cpnn
from chemprop.nn.metrics import ChempropMetric
from lightning import pytorch as pl

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.assembly import ASSAYS, RESPONSE_ASSAY, encode_conditions, n_condition_columns
from cypshift.features.embeddings import embedding_block, embedding_width
from cypshift.features.ionization import block_for as ionization_block_for
from cypshift.features.molecular import featurise
from cypshift.models.base import PredictiveDistribution
from cypshift.models.shared import SCORED_ASSAY

logger = logging.getLogger(__name__)

#: Label columns: the value, then the two carrier columns holding its bounds, and an
#: optional fourth column for the qHTS max-response auxiliary head. Per-row routing is by
#: NaN: a pIC50 row leaves RESPONSE NaN, a response row leaves VALUE/LOWER/UPPER NaN, and
#: chemprop's finiteness mask zeroes the irrelevant task, so the two scales never mix.
VALUE, LOWER, UPPER, RESPONSE = 0, 1, 2, 3
N_LABEL_COLUMNS = 3

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
    qHTS max-response auxiliary head. Each row carries only one of the two (the other is
    NaN upstream, so chemprop's finiteness mask zeroes it), so the losses never mix.
    """

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
        if targets.shape[1] > RESPONSE:
            loss[:, RESPONSE] = (preds[:, RESPONSE] - targets[:, RESPONSE]) ** 2
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
        ensemble: str = "reseed",
        pretraining: str = "none",
        pretraining_cache: str = "data/external/chemeleon/chemeleon_mp.pt",
        auxiliary_response: bool = False,
        physchem: bool = False,
        mechanism: bool = False,
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
        seed: int = 0,
    ) -> None:
        self.endpoints = endpoints
        self.targets = targets
        # The response head is trained on its own assay, so it joins the conditioning
        # vocabulary when it is on (and the label matrix grows a fourth column for it).
        self.auxiliary_response = auxiliary_response
        self.assays = (*assays, RESPONSE_ASSAY) if auxiliary_response else assays
        self._n_label = N_LABEL_COLUMNS + 1 if auxiliary_response else N_LABEL_COLUMNS
        self.conditioning = conditioning
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_members = n_members
        self.loss = loss
        #: How ensemble members differ ("reseed" or "bagging") and which pretrained encoder
        #: to initialise from ("none"/"chemeleon"/"mammal").
        self.ensemble = ensemble
        self.pretraining = pretraining
        self.pretraining_cache = pretraining_cache
        self.seed = seed
        self._members: list[cpmodels.MPNN] = []
        self._n_conditioning = n_condition_columns(conditioning, targets, self.assays)
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
        #: Separate standardisation for the response task, so its 0-1 scale is not forced
        #: onto the pIC50 centre/spread.
        self._resp_centre = 0.0
        self._resp_spread = 1.0

    def _datapoints(self, frame: pd.DataFrame, labelled: bool) -> list[cpdata.MoleculeDatapoint]:
        conditioning = encode_conditions(frame, self.conditioning, self.targets, self.assays)
        smiles = frame["SMILES"].tolist()
        if self._embedding is not None:
            conditioning = np.hstack([conditioning, self._embedding(smiles)])
        if self._descriptor is not None:
            conditioning = np.hstack([conditioning, self._descriptor(smiles)])
        points = []
        for row, (_, record) in enumerate(frame.iterrows()):
            if labelled:
                y = np.full(self._n_label, np.nan)
                if self.auxiliary_response and record["assay"] == RESPONSE_ASSAY:
                    # A response row: fill only the response task, leave pIC50 tasks NaN.
                    y[RESPONSE] = (float(record["value"]) - self._resp_centre) / self._resp_spread
                else:
                    value = float(record["value"])
                    lower = float(record["conf_low"]) if np.isfinite(record["conf_low"]) else value
                    upper = (
                        float(record["conf_high"]) if np.isfinite(record["conf_high"]) else value
                    )
                    if bool(record.get("is_censored", False)):
                        lower = -1e4
                    y[VALUE : UPPER + 1] = (
                        np.array([value, lower, upper], dtype=float) - self._centre
                    ) / self._spread
                weight = float(record["weight"])
            else:
                y = np.zeros(self._n_label)
                weight = 1.0
            points.append(
                cpdata.MoleculeDatapoint.from_smi(
                    record["SMILES"], y=y, x_d=conditioning[row].astype(float), weight=weight
                )
            )
        return points

    def _build(self) -> cpmodels.MPNN:
        if self.pretraining == "chemeleon":
            message_passing = chemeleon_message_passing(Path(self.pretraining_cache))
        else:
            message_passing = cpnn.BondMessagePassing(
                d_h=self.hidden, depth=self.depth, dropout=self.dropout
            )
        # Value task on, carriers off; the response task (when present) on.
        task_weights = [1.0, 0.0, 0.0] + ([1.0] if self.auxiliary_response else [])
        criterion = (
            SoftThresholdLoss(task_weights=torch.tensor([task_weights]))
            if self.loss == "soft_threshold"
            else None
        )
        predictor = cpnn.RegressionFFN(
            n_tasks=self._n_label,
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
        # The pIC50 rows set one centre/spread for the value and its two bound columns;
        # the response rows (a different 0-1 scale) are standardised separately.
        is_response = (
            usable["assay"] == RESPONSE_ASSAY
            if self.auxiliary_response
            else np.zeros(len(usable), dtype=bool)
        )
        pic50 = usable.loc[~is_response, "value"].to_numpy(float)
        self._centre = float(pic50.mean())
        self._spread = float(pic50.std()) or 1.0
        if is_response.any():
            response = usable.loc[is_response, "value"].to_numpy(float)
            self._resp_centre = float(response.mean())
            self._resp_spread = float(response.std()) or 1.0
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
            loader = cpdata.build_dataloader(dataset, batch_size=self.batch_size, num_workers=0)
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
