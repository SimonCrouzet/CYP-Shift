"""Predicting the dose-response curve, and reading every answer off it.

The experiment measures a curve, not a number. A twelve-point run fits inhibition
against concentration and reports the midpoint as an IC50; the single-concentration
screen is *the same curve sampled once*, at 49.5 uM. Modelling the curve makes that
relationship explicit instead of routing the screen through a calibration:

    inhibition(c) = Emax / (1 + 10^(h * (pIC50 - pC)))        pC = -log10(c)

with ``h`` the Hill slope and ``Emax`` the maximal effect. The model predicts those
three parameters, and each kind of observation constrains whatever part of the curve
it actually saw.

* A fitted **pIC50** with its credible interval constrains the midpoint directly.
* A fitted **Emax** constrains the plateau.
* A **screen** reading constrains ``inhibition(49.5 uM)`` -- one point on the curve.
  Nothing is calibrated, nothing is invented, and a compound whose curve never rises
  is simply one whose predicted response at that concentration is near zero.

Preincubation is a second curve rather than an offset on the first, since it is a
different experiment: enzyme destroyed by turnover, not enzyme occupied. Time
dependence is then whatever the two curves say it is, evaluated through the
organisers' own rule.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.conditioning import describe_all
from cypshift.features.assembly import molecular_feature_matrix
from cypshift.features.scaling import BlockScaler
from cypshift.models.base import (
    PredictiveDistribution,
    active_stages,
    rows_for_stage,
    training_stages,
)
from cypshift.models.tdi import TDI_FLOOR_DEFAULT, TDI_SHIFT_DEFAULT

logger = logging.getLogger(__name__)

#: Curve parameters, in the order the network emits them.
PIC50, HILL, EMAX = 0, 1, 2
N_PARAMETERS = 3

#: The single concentration the primary screen was run at, as a pIC50-comparable
#: quantity: -log10(4.95e-5 M).
SCREEN_PC = 4.305

#: Arms. Each is its own curve, not a displacement of the other.
ASSAY_CONDITIONS = ("direct", "preincubated")

#: Assays measured under the preincubation (TDI) condition: the TDI dose-response and the
#: primary single-concentration screen, which the organisers ran under preincubation to catch
#: as many hits as possible. Every other assay (the direct DRC and all auxiliary assays) is
#: direct, so this is the complete preincubated set.
_PREINCUBATED_ASSAYS = ("tdi_drc", "direct_inhibition_screen")


def assay_condition_index(assays: "pd.Series") -> np.ndarray:
    """Map each row's assay to its curve condition index: 1 preincubated, 0 direct.

    ``ASSAY_CONDITIONS`` orders direct first, preincubated second, so a preincubated assay maps
    to 1. The screen is preincubated, so it constrains the same curve the TDI read compares
    against the direct arm -- not the direct curve itself.
    """
    return assays.isin(_PREINCUBATED_ASSAYS).to_numpy(int)


#: How the network is made to depend on which enzyme and assay_condition a row came from.
CONDITIONING = ("heads", "descriptor", "film")

#: The strategies needing a descriptor row per enzyme.
DESCRIBED_CONDITIONING = frozenset({"descriptor", "film"})

#: Offsets chosen so a raw output of zero gives the pharmacological default: a Hill
#: slope of 1, which is simple mass-action binding, and near-complete maximal effect.
#:
#: This matters because the curve's *shape* is only identifiable where a dose-response
#: point constrains it. A row reporting a fitted midpoint says nothing about the slope,
#: so without an anchored default the slope drifts to wherever initialisation left it
#: and the curve flattens -- which decouples the predicted response at a concentration
#: from the predicted potency, defeating the point of modelling a curve at all.
HILL_OFFSET = 0.5413  # softplus(0.5413) == 1.0
EMAX_OFFSET = 2.2  # sigmoid(2.2) ~= 0.9
_LN10 = math.log(10.0)  # convert the base-10 Hill exponent into a stable sigmoid logit


def inhibition_at(parameters: torch.Tensor, pc: float | torch.Tensor) -> torch.Tensor:
    """Fraction inhibited at concentration ``pc``, from Hill parameters.

    Args:
        parameters: Shape ``(..., 3)`` holding raw pIC50, Hill and Emax outputs. Hill
            is passed through softplus to keep the curve monotone, and Emax through a
            sigmoid to keep the plateau a fraction. Both carry an offset so that an
            unconstrained parameter falls back to the pharmacological default rather
            than to whatever initialisation produced.
        pc: ``-log10`` of the concentration.

    Returns:
        Inhibited fraction in ``[0, 1]``, same leading shape as ``parameters``.
    """
    pic50 = parameters[..., PIC50]
    hill = nn.functional.softplus(parameters[..., HILL] + HILL_OFFSET) + 1e-3
    emax = torch.sigmoid(parameters[..., EMAX] + EMAX_OFFSET)
    # emax / (1 + 10**(hill*(pc - pic50))) rewritten as a sigmoid: torch.pow(10, .)
    # overflows to inf for a large exponent and leaves NaN gradients even on rows whose
    # loss never uses this branch, whereas sigmoid is numerically stable everywhere.
    return emax * torch.sigmoid(_LN10 * hill * (pic50 - pc))


def build_curve_trunk(n_features: int, hidden: int, dropout: float, depth: int = 2) -> nn.Module:
    """The default curve backbone: an MLP from features to a ``hidden // 2`` representation.

    ``depth`` is the number of hidden layers (>= 1). The first maps to ``hidden``; the last maps
    to ``hidden // 2`` (the width the dose-response head reads); any in between stay at
    ``hidden``. Deepening the backbone is the cheapest "different arch at the end" lever.
    """
    if depth < 1:
        raise ValueError(f"curve trunk depth must be >= 1, got {depth}")
    widths = [n_features] + [hidden] * (depth - 1) + [hidden // 2]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(widths[:-1], widths[1:], strict=True):
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class CurveNetwork(nn.Module):
    """Shared trunk over molecular features, emitting curve parameters per enzyme and
    assay_condition.

    Two ways to make the output depend on which enzyme and assay_condition a row belongs to, and
    which is better is an empirical question rather than a settled one:

    ``heads``
        One output head per (enzyme, assay_condition). Maximally flexible, and the variant with
        industrial-scale validation in multi-task property prediction. But a head is an
        index: an enzyme outside the scored set trains parameters that are thrown away,
        so nothing pretrains.

    ``descriptor``
        One shared head, given a vector describing the enzyme's pocket. Less free per
        enzyme, but it can learn transferable rules -- an acidic anchor prefers a basic
        ligand -- so an auxiliary enzyme's data improves the scored ones. This is what
        makes pretraining across enzymes possible.

    ``film``
        The same descriptor, but multiplying the molecular representation instead of
        being concatenated to it.

    The difference between the last two decides whether conditioning can express
    anything interesting, and it is easy to miss. Concatenating into a linear head
    computes ``W1 @ shared + W2 @ described``: the enzyme term does not involve the
    molecule at all, so the whole effect of knowing which enzyme it is amounts to a
    constant offset on the curve parameters. It cannot say that an acid helps at CYP2C9
    and hurts at CYP2D6, because that is a statement about an interaction.

    Feature-wise linear modulation (Perez et al., AAAI 2018) makes the descriptor
    produce a scale and a shift applied to the shared representation, so the enzyme
    multiplies the molecular features. That product is the cross-term the
    proteochemometrics literature identifies as where transfer signal actually lives
    (Qiu et al., Brief Bioinform 2017;18:125). The modulation is initialised to the
    identity, so training starts from exactly the unconditioned model and has to earn
    any departure from it.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        n_assay_conditions: int = 2,
        hidden: int = 256,
        dropout: float = 0.2,
        conditioning: str = "heads",
        descriptors: torch.Tensor | None = None,
        trunk: nn.Module | None = None,
        trunk_output_dim: int | None = None,
        trunk_depth: int = 2,
    ) -> None:
        super().__init__()
        if conditioning not in CONDITIONING:
            raise ValueError(
                f"unknown conditioning {conditioning!r}; expected one of {CONDITIONING}"
            )
        if conditioning in DESCRIBED_CONDITIONING and descriptors is None:
            raise ValueError(f"{conditioning} conditioning needs one descriptor row per enzyme")

        self.n_targets, self.n_assay_conditions, self.conditioning = (
            n_targets,
            n_assay_conditions,
            conditioning,
        )
        # The curve is a backbone that maps features to a shared representation, then a
        # dose-response head on top. The backbone is pluggable: pass ``trunk`` (with its output
        # width ``trunk_output_dim``) to read any representation -- a deeper MLP, a pretrained
        # encoder, a shared backbone from the best config -- and the head logic below is
        # unchanged. The default is the built-in feature MLP.
        if trunk is not None:
            if trunk_output_dim is None:
                raise ValueError("an injected trunk needs its output width (trunk_output_dim)")
            self.trunk = trunk
            width = trunk_output_dim
        else:
            self.trunk = build_curve_trunk(n_features, hidden, dropout, trunk_depth)
            width = hidden // 2
        if conditioning == "heads":
            self.heads = nn.ModuleList(
                [nn.Linear(width, N_PARAMETERS) for _ in range(n_targets * n_assay_conditions)]
            )
        elif conditioning == "descriptor":
            self.register_buffer("descriptors", descriptors)
            self.head = nn.Linear(width + descriptors.shape[1] + n_assay_conditions, N_PARAMETERS)
        else:
            self.register_buffer("descriptors", descriptors)
            self.modulation = nn.Linear(descriptors.shape[1] + n_assay_conditions, 2 * width)
            # Zeroed so the layer emits no scale and no shift at the start. With the
            # +1 in forward() that makes the modulation the identity, so training
            # begins from the unconditioned model rather than from noise injected into
            # every feature.
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
            self.head = nn.Linear(width, N_PARAMETERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``(batch, n_targets, n_assay_conditions, 3)``."""
        shared = self.trunk(x)
        if self.conditioning == "heads":
            stacked = torch.stack([head(shared) for head in self.heads], dim=1)
            return stacked.view(-1, self.n_targets, self.n_assay_conditions, N_PARAMETERS)

        # One shared head, evaluated once per (enzyme, assay_condition).
        outputs = []
        for target in range(self.n_targets):
            for assay_condition in range(self.n_assay_conditions):
                assay_condition_code = torch.zeros(
                    len(shared), self.n_assay_conditions, device=shared.device
                )
                assay_condition_code[:, assay_condition] = 1.0
                described = self.descriptors[target].expand(len(shared), -1)
                condition = torch.cat([described, assay_condition_code], dim=1)
                if self.conditioning == "descriptor":
                    outputs.append(self.head(torch.cat([shared, condition], dim=1)))
                else:
                    scale, shift = self.modulation(condition).chunk(2, dim=1)
                    outputs.append(self.head((1.0 + scale) * shared + shift))
        return torch.stack(outputs, dim=1).view(
            -1, self.n_targets, self.n_assay_conditions, N_PARAMETERS
        )


def curve_loss(
    parameters: torch.Tensor,
    observation_kind: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    screen_pc: float = SCREEN_PC,
    tau: float = 0.5,
) -> torch.Tensor:
    """Soft-threshold error against whichever part of the curve a row observed.

    Args:
        parameters: Per-row curve parameters, shape ``(n, 3)``.
        observation_kind: 0 for a fitted pIC50, 1 for a screen reading.
        lower: Lower bound of the observed quantity.
        upper: Upper bound.
        weight: Per-row trust.
        screen_pc: Concentration the screen was run at.
        tau: Asymmetry. At 0.5 the dead-zone penalty is symmetric; ``tau>0.5`` penalises
            falling below the band (under-prediction) more than overshooting it, the upward
            bias the metric rewards -- the curve analogue of the expectile/pinball losses.

    Returns:
        Weighted mean loss.
    """
    midpoint = parameters[:, PIC50]
    response = inhibition_at(parameters, screen_pc)
    predicted = torch.where(observation_kind == 0, midpoint, response)
    above = torch.clamp(predicted - upper, min=0.0)
    below = torch.clamp(lower - predicted, min=0.0)
    penalty = 2.0 * (1.0 - tau) * above + 2.0 * tau * below
    return (penalty * weight).sum() / weight.sum().clamp(min=1e-8)


class _CurveReadout:
    """The ways of reading an answer off a fitted curve, shared by every curve backbone.

    A backbone (the fingerprint MLP, or the D-MPNN graph encoder) supplies ``_curves`` --
    the per-pass curve parameters, shape ``(passes, compounds, targets, assay_conditions, 3)``
    -- and ``targets``/``tdi_*`` attributes; the regression midpoint, the response at any
    concentration, and the is_TDI call are the same read regardless of which backbone produced
    the curve, so they live here once.
    """

    def predict(self, smiles: list[str]) -> PredictiveDistribution:
        """The regression track's prediction, satisfying the shared model interface.

        The midpoint of the direct assay_condition's curve, one column per enzyme, with the
        Monte Carlo dropout passes kept as the member axis so the spread reaches the
        interval machinery rather than being averaged away here.

        This exists because :class:`cypshift.models.base.Model` requires it and the
        pipeline calls it. Without it the curve model could be built and fitted but
        never scored -- every run through the pipeline died at prediction.
        """
        curves = self._curves(smiles)
        direct = curves[:, :, :, ASSAY_CONDITIONS.index("direct"), PIC50]
        # self.targets may be wider than the scored four when auxiliary enzymes were
        # trained on; select the scored endpoints in their canonical order.
        columns = [self.targets.index(e.split("_")[0]) for e in REGRESSION_ENDPOINTS]
        return PredictiveDistribution(
            samples=direct[:, :, columns], endpoints=tuple(REGRESSION_ENDPOINTS)
        )

    def predict_pic50(
        self, smiles: list[str], assay_condition: str = "direct"
    ) -> NDArray[np.float64]:
        """The curve's midpoint, which is what the regression track asks for."""
        return self._curves(smiles).mean(axis=0)[
            :, :, ASSAY_CONDITIONS.index(assay_condition), PIC50
        ]

    def predict_response(
        self, smiles: list[str], pc: float = SCREEN_PC, assay_condition: str = "direct"
    ) -> NDArray[np.float64]:
        """Inhibited fraction at any concentration, read off the predicted curve."""
        curves = torch.from_numpy(
            self._curves(smiles).mean(axis=0)[:, :, ASSAY_CONDITIONS.index(assay_condition)]
        )
        return inhibition_at(curves, pc).numpy()

    def predict_is_time_dependent(self, smiles: list[str]) -> NDArray[np.bool_]:
        """Call is_TDI from the two assay_conditions' curves, using the model's decision
        thresholds."""
        floor, shift = self.tdi_assay_floor, self.tdi_shift_threshold
        curves = self._curves(smiles).mean(axis=0)
        direct = curves[:, :, ASSAY_CONDITIONS.index("direct"), PIC50]
        preincubated = curves[:, :, ASSAY_CONDITIONS.index("preincubated"), PIC50]
        return np.where(
            direct > floor,
            (preincubated - direct) > shift,
            preincubated > floor + shift,
        )


class CurveModel(_CurveReadout):
    """Predicts a dose-response curve per compound, enzyme and assay_condition."""

    name = "curve"
    #: Required by :class:`cypshift.models.base.Model`. The curve model also predicts
    #: the TDI endpoints, but those are scored on their own track.
    endpoints = tuple(REGRESSION_ENDPOINTS)

    def __init__(
        self,
        targets: tuple[str, ...] = ISOFORMS,
        n_bits: int = 2048,
        use_fingerprint: bool = True,
        physchem: bool = True,
        mechanism: bool = True,
        fingerprint_radius: int = 2,
        screen_model: object | None = None,
        hidden: int = 256,
        dropout: float = 0.2,
        trunk_depth: int = 2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        n_forward_passes: int = 30,
        conditioning: str = "heads",
        descriptor_sources: tuple[str, ...] = ("curated",),
        tdi_shift_threshold: float = TDI_SHIFT_DEFAULT,
        tdi_assay_floor: float = TDI_FLOOR_DEFAULT,
        loss_tau: float = 0.5,
        seed: int = 0,
        training: object | None = None,
        molecular_embedding: str = "none",
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
    ) -> None:
        self.conditioning = conditioning
        # The model's own is_TDI decision thresholds, applied to its (noisy) predicted
        # assay_condition shift. They default to the organisers' label rule (a 2-fold, 0.301-log
        # shift
        # above a 4.0 assay floor) but are swept, since the threshold that best recovers the
        # labels from predicted shifts need not equal the one that defines them.
        self.tdi_shift_threshold = tdi_shift_threshold
        self.tdi_assay_floor = tdi_assay_floor
        # Asymmetry of the dose-response dead-zone loss; >0.5 errs high to resist compression.
        self.loss_tau = loss_tau
        #: Which sources build the per-enzyme vector when conditioning is
        #: "descriptor" or "film"; see
        #: :data:`cypshift.data.conditioning.DESCRIPTOR_SOURCES`.
        self.descriptor_sources = descriptor_sources
        self.targets = targets
        self.n_bits = n_bits
        #: Which molecular feature blocks to include and the fingerprint radius.
        #: These are read by featurise; toggling physchem changes the dense-block width,
        #: which the scaler follows. Swept via features.* in the config.
        self.use_fingerprint = use_fingerprint
        self.physchem = physchem
        self.mechanism = mechanism
        self.fingerprint_radius = fingerprint_radius
        #: Optional callable appended to the feature matrix, used to feed the
        #: single-concentration screen in as an input rather than as extra training
        #: rows. See :mod:`cypshift.data.screen_model`.
        self.screen_model = screen_model
        self.hidden = hidden
        self.dropout = dropout
        #: Hidden-layer count of the default curve backbone; deepen it for a larger-capacity
        #: backbone under the same dose-response head.
        self.trunk_depth = trunk_depth
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_forward_passes = n_forward_passes
        self.seed = seed
        #: Training curriculum (an object exposing ``.stages``), or None for one pooled
        #: stage over every data_space.
        self.training = training
        #: Which cached foundation-model embedding to append, or "none".
        self.molecular_embedding = molecular_embedding
        #: Append the computed ionization / conformer-geometry block, and where it caches.
        self.ionization_geometry = ionization_geometry
        self.ionization_cache = ionization_cache
        self._model: CurveNetwork | None = None
        self._scaler: BlockScaler | None = None

    def _features(self, smiles: list[str], fit: bool) -> torch.Tensor:
        matrix = molecular_feature_matrix(
            smiles,
            n_bits=self.n_bits,
            fingerprint_radius=self.fingerprint_radius,
            use_fingerprint=self.use_fingerprint,
            physchem=self.physchem,
            mechanism=self.mechanism,
            screen_model=self.screen_model,
            ionization_geometry=self.ionization_geometry,
            ionization_cache=self.ionization_cache,
            molecular_embedding=self.molecular_embedding,
        ).astype(np.float32)
        if fit:
            # Count block = the Morgan fingerprint (absent when use_fingerprint is off); every
            # column after it is a dense descriptor to standardise (physchem, mechanistic,
            # screen_model, ionization, embedding), not physchem alone.
            n_count = self.n_bits if self.use_fingerprint else 0
            self._scaler = BlockScaler(n_count, matrix.shape[1] - n_count).fit(matrix)
        if self._scaler is None:
            raise RuntimeError("model is not fitted")
        return torch.from_numpy(self._scaler.transform(matrix))

    def fit(self, observations: pd.DataFrame) -> "CurveModel":
        """Train on the observation table.

        Screen rows are recognised by their assay and constrain the curve at the screen
        concentration; every other row constrains the curve's midpoint.
        """
        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        molecules = usable["SMILES"].drop_duplicates().tolist()
        row_of = {smiles: index for index, smiles in enumerate(molecules)}
        features = self._features(molecules, fit=True)

        value = usable["value"].to_numpy(float)
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -1e4, lower)

        is_screen = (usable["assay"] == "direct_inhibition_screen").to_numpy()
        molecule_index = torch.tensor([row_of[s] for s in usable["SMILES"]], dtype=torch.long)
        target_index = torch.tensor(
            [self.targets.index(t) for t in usable["target"]], dtype=torch.long
        )
        # The curve has exactly two assay conditions: preincubated and direct. The preincubated
        # arm carries the TDI dose-response and the primary single-concentration screen (run
        # under preincubation); the direct curves and every auxiliary assay (octant, veith/tox21,
        # chembl/sair, novartis) map to the direct condition, so auxiliary rows enter without a
        # per-assay offset of their own. Curve-with-auxiliary is therefore a different
        # measurement-model assumption than the per-assay conditioning the graph and boosted
        # paths apply, not the same auxiliary conditioning.
        condition_index = torch.tensor(assay_condition_index(usable["assay"]), dtype=torch.long)
        kind = torch.tensor(is_screen.astype(int), dtype=torch.long)
        lower_t = torch.tensor(lower, dtype=torch.float32)
        upper_t = torch.tensor(upper, dtype=torch.float32)
        weight_t = torch.tensor(usable["weight"].to_numpy(float), dtype=torch.float32)

        descriptors = None
        # Both described modes need the vector, not just "descriptor" -- film consumes
        # the same rows and merely applies them multiplicatively.
        if self.conditioning in DESCRIBED_CONDITIONING:
            descriptors = torch.tensor(
                describe_all(list(self.targets), tuple(self.descriptor_sources)),
                dtype=torch.float32,
            )
        self._model = CurveNetwork(
            features.shape[1],
            len(self.targets),
            len(ASSAY_CONDITIONS),
            self.hidden,
            self.dropout,
            conditioning=self.conditioning,
            descriptors=descriptors,
            trunk_depth=self.trunk_depth,
        )
        logger.info(
            "training curves on %d observations, %d of them screen readings",
            len(usable),
            int(is_screen.sum()),
        )

        def optimise(stage_rows: torch.Tensor, epochs: int, lr: float, freeze_trunk: bool) -> None:
            """Train the current curve network on a subset of rows, carrying weights forward."""
            if len(stage_rows) == 0:
                return
            if freeze_trunk:
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(False)
            trainable = [p for p in self._model.parameters() if p.requires_grad]
            if not trainable:
                logger.info("stage frozen with no trainable head; skipping")
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(True)
                return
            optimiser = torch.optim.Adam(trainable, lr=lr)
            m = len(stage_rows)
            for epoch in range(epochs):
                order = stage_rows[torch.randperm(m)]
                total = 0.0
                for start in range(0, m, self.batch_size):
                    batch = order[start : start + self.batch_size]
                    optimiser.zero_grad()
                    predicted = self._model(features[molecule_index[batch]])
                    rows = torch.arange(len(batch))
                    parameters = predicted[rows, target_index[batch], condition_index[batch]]
                    loss = curve_loss(
                        parameters,
                        kind[batch],
                        lower_t[batch],
                        upper_t[batch],
                        weight_t[batch],
                        tau=self.loss_tau,
                    )
                    loss.backward()
                    optimiser.step()
                    total += loss.detach().item() * len(batch)
                if epoch % 20 == 0 or epoch == epochs - 1:
                    logger.info("  epoch %2d  loss %.4f", epoch, total / m)
            if freeze_trunk:
                for parameter in self._model.trunk.parameters():
                    parameter.requires_grad_(True)

        all_rows = torch.arange(len(usable))
        stages = training_stages(self.training, self.epochs, self.learning_rate)
        for stage in active_stages(usable, stages):
            mask = rows_for_stage(usable, stage)
            logger.info(
                "stage data_spaces=%s assays=%s: %d rows, %d epochs, lr %.1e%s",
                stage.data_spaces,
                stage.assays or "all",
                int(mask.sum()),
                stage.epochs,
                stage.lr,
                " (trunk frozen)" if stage.freeze_trunk else "",
            )
            optimise(all_rows[torch.tensor(mask)], stage.epochs, stage.lr, stage.freeze_trunk)
        return self

    def _curves(self, smiles: list[str]) -> NDArray[np.float64]:
        """Curve parameters, shape ``(passes, compounds, targets, assay_conditions, 3)``."""
        if self._model is None:
            raise RuntimeError("model is not fitted")
        self._model.train()
        torch.manual_seed(self.seed)
        features = self._features(smiles, fit=False)
        with torch.no_grad():
            return np.stack([self._model(features).numpy() for _ in range(self.n_forward_passes)])

    # predict / predict_pic50 / predict_response / predict_is_time_dependent are inherited
    # from _CurveReadout, shared with the graph-backbone curve.


class _GraphTrunk(nn.Module):
    """A chemprop D-MPNN encoder adapted as a curve backbone.

    ``CurveNetwork`` calls ``self.trunk(x)`` and uses only its output, so a trunk may take
    anything as ``x`` provided it returns ``(batch, width)``. This one takes a chemprop batch:
    it runs message passing over the molecular graph, mean-aggregates to one vector per
    molecule, and (when present) concatenates the standardised dense descriptor block the batch
    carries in ``x_d``. The dose-response head above it is unchanged, so the curve is read off a
    representation learned from the molecular graph instead of the fingerprint MLP's output.

    A dropout is applied to that representation before the head reads it. ``CurveNetwork`` adds
    no dropout after an injected trunk and its heads are linear, and a frozen (or zero-dropout
    pretrained) encoder applies none of its own, so without this the configured dropout would be
    ignored and the Monte Carlo prediction passes would be deterministic -- repeating one
    computation instead of estimating a spread. Placing it here makes the readout dropout the one
    guaranteed Monte Carlo source regardless of the encoder's state.
    """

    def __init__(
        self,
        message_passing: nn.Module,
        aggregation: nn.Module,
        xd_width: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.message_passing = message_passing
        self.aggregation = aggregation
        self.readout_dropout = nn.Dropout(dropout)
        #: The representation width the curve head reads: encoder output plus any x_d descriptors.
        self.output_dim = message_passing.output_dim + xd_width

    def forward(self, batch: object) -> torch.Tensor:
        node = self.message_passing(batch.bmg, batch.V_d)
        pooled = self.aggregation(node, batch.bmg.batch)
        if batch.X_d is not None:
            pooled = torch.cat([pooled, batch.X_d.to(pooled.device)], dim=1)
        return self.readout_dropout(pooled)


class GraphCurveModel(_CurveReadout):
    """The dose-response curve head on a chemprop D-MPNN backbone.

    The fingerprint-MLP curve (:class:`CurveModel`) and this one share the dose-response head
    and every way of reading it (:class:`_CurveReadout`); they differ only in the backbone that
    produces the shared representation. Here a learned D-MPNN encoder -- optionally initialised
    from the same CheMeleon pretraining the graph model uses -- replaces the fingerprint MLP, so
    the curve is read off a representation learned from the molecular graph. This is the
    cross-model track: the curve's two reads (direct for regression, direct-vs-preincubated for
    TDI) sitting on the graph family's backbone rather than the curve model's own MLP.

    The screen enters as observation rows constraining the curve at the screen concentration,
    exactly as for the MLP curve; the enzyme/assay_condition conditioning is the curve head's
    (heads/descriptor/film), not the graph model's readout conditioning. The learned encoder
    replaces the fingerprint, so the fingerprint feature flags do not apply; physchem/mechanism/
    ionization descriptors still ride along as chemprop ``x_d``, standardised on training rows.
    """

    name = "curve_graph"
    #: Required by :class:`cypshift.models.base.Model`; the TDI endpoints are scored separately.
    endpoints = tuple(REGRESSION_ENDPOINTS)

    def __init__(
        self,
        targets: tuple[str, ...] = ISOFORMS,
        hidden: int = 300,
        depth: int = 3,
        dropout: float = 0.1,
        pretraining: str = "none",
        pretraining_cache: str = "data/external/chemeleon/chemeleon_mp.pt",
        pretraining_finetune: bool = True,
        physchem: bool = False,
        mechanism: bool = False,
        ionization_geometry: bool = False,
        ionization_cache: str = "data/external/ionization/features.parquet",
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        num_workers: int = 0,
        n_forward_passes: int = 30,
        conditioning: str = "heads",
        descriptor_sources: tuple[str, ...] = ("curated",),
        tdi_shift_threshold: float = TDI_SHIFT_DEFAULT,
        tdi_assay_floor: float = TDI_FLOOR_DEFAULT,
        loss_tau: float = 0.5,
        seed: int = 0,
    ) -> None:
        if conditioning not in CONDITIONING:
            raise ValueError(
                f"unknown conditioning {conditioning!r}; expected one of {CONDITIONING}"
            )
        self.targets = targets
        #: Encoder message-passing width (d_h) and depth; dropout is shared with the head.
        self.hidden = hidden
        self.depth = depth
        self.dropout = dropout
        #: Which pretrained encoder initialises the backbone ("none"/"chemeleon") and whether it
        #: is fine-tuned or frozen as a fixed featuriser.
        self.pretraining = pretraining
        self.pretraining_cache = pretraining_cache
        self.pretraining_finetune = pretraining_finetune
        #: Dense descriptor blocks appended to the pooled graph representation through x_d.
        self.physchem = physchem
        self.mechanism = mechanism
        self.ionization_geometry = ionization_geometry
        self.ionization_cache = ionization_cache
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.n_forward_passes = n_forward_passes
        self.conditioning = conditioning
        self.descriptor_sources = descriptor_sources
        self.tdi_shift_threshold = tdi_shift_threshold
        self.tdi_assay_floor = tdi_assay_floor
        self.loss_tau = loss_tau
        self.seed = seed
        self._model: CurveNetwork | None = None
        self._descriptor = None
        self._xd_centre: np.ndarray | None = None
        self._xd_scale: np.ndarray | None = None
        self._device = "cpu"

    def _descriptor_matrix(self, smiles: list[str]) -> np.ndarray:
        """The dense descriptor block (physchem/mechanism/ionization) for the x_d slot, or empty."""
        if self._descriptor is None:
            return np.empty((len(smiles), 0), dtype=np.float32)
        return self._descriptor(smiles).astype(np.float32)

    def _encoder(self) -> nn.Module:
        """The D-MPNN message-passing encoder: CheMeleon-pretrained or freshly initialised."""
        from chemprop import nn as cpnn

        from cypshift.models.graph import chemeleon_message_passing

        if self.pretraining == "chemeleon":
            message_passing = chemeleon_message_passing(Path(self.pretraining_cache))
            # The checkpoint carries its own dropout (the local CheMeleon encoder's is 0.0); set
            # the configured dropout so a fine-tuned encoder is regularised. A frozen encoder's
            # dropout is inert, but _GraphTrunk's readout dropout still supplies the MC spread.
            if isinstance(getattr(message_passing, "dropout", None), nn.Dropout):
                message_passing.dropout.p = self.dropout
            if not self.pretraining_finetune:
                for parameter in message_passing.parameters():
                    parameter.requires_grad = False
            return message_passing
        return cpnn.BondMessagePassing(d_h=self.hidden, depth=self.depth, dropout=self.dropout)

    def fit(self, observations: pd.DataFrame) -> "GraphCurveModel":
        """Train the curve head on a D-MPNN backbone over the observation table.

        Screen rows constrain the curve at the screen concentration; every other row constrains
        the midpoint. One datapoint per observation, the per-row routing carried in ``y`` as
        ``[target_index, assay_condition_index, kind, lower, upper, weight]`` and read back in
        the loop, so the custom dose-response loss selects the right (enzyme, condition) curve.
        """
        from chemprop import data as cpdata

        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        if self.physchem or self.mechanism or self.ionization_geometry:
            from cypshift.features.ionization import block_for as ionization_block_for
            from cypshift.features.molecular import featurise

            ionization = ionization_block_for(self.ionization_geometry, Path(self.ionization_cache))

            def descriptor_block(smiles: list[str]) -> np.ndarray:
                return featurise(
                    smiles,
                    fingerprint=False,
                    physchem=self.physchem,
                    mechanism=self.mechanism,
                    ionization=ionization,
                )

            self._descriptor = descriptor_block

        value = usable["value"].to_numpy(float)
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -1e4, lower)
        is_screen = (usable["assay"] == "direct_inhibition_screen").to_numpy()
        target_index = np.array([self.targets.index(t) for t in usable["target"]], dtype=float)
        condition_index = assay_condition_index(usable["assay"]).astype(float)
        kind = is_screen.astype(float)
        weight = usable["weight"].to_numpy(float)

        descriptors = self._descriptor_matrix(usable["SMILES"].tolist())
        xd_width = descriptors.shape[1]
        if xd_width:
            self._xd_centre = descriptors.mean(axis=0)
            self._xd_scale = descriptors.std(axis=0) + 1e-6
            descriptors = (descriptors - self._xd_centre) / self._xd_scale

        condition_descriptors = None
        if self.conditioning in DESCRIBED_CONDITIONING:
            condition_descriptors = torch.tensor(
                describe_all(list(self.targets), tuple(self.descriptor_sources)),
                dtype=torch.float32,
            )
        trunk = _GraphTrunk(self._encoder(), self._mean_aggregation(), xd_width, self.dropout)
        self._model = CurveNetwork(
            trunk.output_dim,
            len(self.targets),
            len(ASSAY_CONDITIONS),
            self.hidden,
            self.dropout,
            conditioning=self.conditioning,
            descriptors=condition_descriptors,
            trunk=trunk,
            trunk_output_dim=trunk.output_dim,
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        logger.info(
            "training curve-on-graph on %d observations (%d screen) on %s",
            len(usable),
            int(is_screen.sum()),
            self._device,
        )

        points = []
        for row in range(len(usable)):
            y = np.array(
                [
                    target_index[row],
                    condition_index[row],
                    kind[row],
                    lower[row],
                    upper[row],
                    weight[row],
                ],
                dtype=float,
            )
            x_d = descriptors[row].astype(float) if xd_width else None
            points.append(
                cpdata.MoleculeDatapoint.from_smi(usable["SMILES"].iloc[row], y=y, x_d=x_d)
            )
        loader = cpdata.build_dataloader(
            cpdata.MoleculeDataset(points),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
        )
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(trainable, lr=self.learning_rate)
        for epoch in range(self.epochs):
            self._model.train()
            total, seen = 0.0, 0
            for batch in loader:
                batch.bmg.to(self._device)
                targets_row = batch.Y.to(self._device)
                optimiser.zero_grad()
                predicted = self._model(batch)
                rows = torch.arange(len(targets_row), device=self._device)
                parameters = predicted[rows, targets_row[:, 0].long(), targets_row[:, 1].long()]
                loss = curve_loss(
                    parameters,
                    targets_row[:, 2].long(),
                    targets_row[:, 3],
                    targets_row[:, 4],
                    targets_row[:, 5],
                    tau=self.loss_tau,
                )
                loss.backward()
                optimiser.step()
                total += loss.detach().item() * len(targets_row)
                seen += len(targets_row)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d  loss %.4f", epoch, total / max(seen, 1))
        return self

    @staticmethod
    def _mean_aggregation() -> nn.Module:
        from chemprop import nn as cpnn

        return cpnn.MeanAggregation()

    def _curves(self, smiles: list[str]) -> NDArray[np.float64]:
        """Curve parameters, shape ``(passes, compounds, targets, assay_conditions, 3)``.

        Dropout stays on for the Monte Carlo passes (both the encoder's and the head's), the
        same spread source as the MLP curve, so the interval machinery sees a predictive spread.
        """
        from chemprop import data as cpdata

        if self._model is None:
            raise RuntimeError("model is not fitted")
        descriptors = self._descriptor_matrix(smiles)
        if descriptors.shape[1] and self._xd_centre is not None:
            descriptors = (descriptors - self._xd_centre) / self._xd_scale
        points = [
            cpdata.MoleculeDatapoint.from_smi(
                s,
                y=np.zeros(1, dtype=float),
                x_d=(descriptors[i].astype(float) if descriptors.shape[1] else None),
            )
            for i, s in enumerate(smiles)
        ]
        loader = cpdata.build_dataloader(
            cpdata.MoleculeDataset(points),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
        self._model.train()
        torch.manual_seed(self.seed)
        passes = []
        for _ in range(self.n_forward_passes):
            outputs = []
            for batch in loader:
                batch.bmg.to(self._device)
                with torch.no_grad():
                    outputs.append(self._model(batch).cpu().numpy())
            passes.append(np.concatenate(outputs, axis=0))
        return np.stack(passes)
