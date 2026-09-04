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

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from cypshift.config import ISOFORMS, TDI_SHIFT_THRESHOLD
from cypshift.data.enzymes import DESCRIPTOR_FAMILIES, describe_all
from cypshift.features.molecular import featurise, physicochemical
from cypshift.features.scaling import BlockScaler

logger = logging.getLogger(__name__)

#: Curve parameters, in the order the network emits them.
PIC50, HILL, EMAX = 0, 1, 2
N_PARAMETERS = 3

#: The single concentration the primary screen was run at, as a pIC50-comparable
#: quantity: -log10(4.95e-5 M).
SCREEN_PC = 4.305

#: Arms. Each is its own curve, not a displacement of the other.
ARMS = ("direct", "preincubated")

#: How the network is made to depend on which enzyme and arm a row came from.
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
    return emax / (1.0 + torch.pow(10.0, hill * (pc - pic50)))


class CurveNetwork(nn.Module):
    """Shared trunk over molecular features, emitting curve parameters per enzyme and arm.

    Two ways to make the output depend on which enzyme and arm a row belongs to, and
    which is better is an empirical question rather than a settled one:

    ``heads``
        One output head per (enzyme, arm). Maximally flexible, and the variant with
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
        n_arms: int = 2,
        hidden: int = 256,
        dropout: float = 0.2,
        conditioning: str = "heads",
        descriptors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if conditioning not in CONDITIONING:
            raise ValueError(
                f"unknown conditioning {conditioning!r}; expected one of {CONDITIONING}"
            )
        if conditioning in DESCRIBED_CONDITIONING and descriptors is None:
            raise ValueError(f"{conditioning} conditioning needs one descriptor row per enzyme")

        self.n_targets, self.n_arms, self.conditioning = n_targets, n_arms, conditioning
        self.trunk = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        width = hidden // 2
        if conditioning == "heads":
            self.heads = nn.ModuleList(
                [nn.Linear(width, N_PARAMETERS) for _ in range(n_targets * n_arms)]
            )
        elif conditioning == "descriptor":
            self.register_buffer("descriptors", descriptors)
            self.head = nn.Linear(width + descriptors.shape[1] + n_arms, N_PARAMETERS)
        else:
            self.register_buffer("descriptors", descriptors)
            self.modulation = nn.Linear(descriptors.shape[1] + n_arms, 2 * width)
            # Zeroed so the layer emits no scale and no shift at the start. With the
            # +1 in forward() that makes the modulation the identity, so training
            # begins from the unconditioned model rather than from noise injected into
            # every feature.
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
            self.head = nn.Linear(width, N_PARAMETERS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``(batch, n_targets, n_arms, 3)``."""
        shared = self.trunk(x)
        if self.conditioning == "heads":
            stacked = torch.stack([head(shared) for head in self.heads], dim=1)
            return stacked.view(-1, self.n_targets, self.n_arms, N_PARAMETERS)

        # One shared head, evaluated once per (enzyme, arm).
        outputs = []
        for target in range(self.n_targets):
            for arm in range(self.n_arms):
                arm_code = torch.zeros(len(shared), self.n_arms, device=shared.device)
                arm_code[:, arm] = 1.0
                described = self.descriptors[target].expand(len(shared), -1)
                condition = torch.cat([described, arm_code], dim=1)
                if self.conditioning == "descriptor":
                    outputs.append(self.head(torch.cat([shared, condition], dim=1)))
                else:
                    scale, shift = self.modulation(condition).chunk(2, dim=1)
                    outputs.append(self.head((1.0 + scale) * shared + shift))
        return torch.stack(outputs, dim=1).view(-1, self.n_targets, self.n_arms, N_PARAMETERS)


def curve_loss(
    parameters: torch.Tensor,
    observation_kind: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weight: torch.Tensor,
    screen_pc: float = SCREEN_PC,
) -> torch.Tensor:
    """Soft-threshold error against whichever part of the curve a row observed.

    Args:
        parameters: Per-row curve parameters, shape ``(n, 3)``.
        observation_kind: 0 for a fitted pIC50, 1 for a screen reading.
        lower: Lower bound of the observed quantity.
        upper: Upper bound.
        weight: Per-row trust.
        screen_pc: Concentration the screen was run at.

    Returns:
        Weighted mean loss.
    """
    midpoint = parameters[:, PIC50]
    response = inhibition_at(parameters, screen_pc)
    predicted = torch.where(observation_kind == 0, midpoint, response)
    above = torch.clamp(predicted - upper, min=0.0)
    below = torch.clamp(lower - predicted, min=0.0)
    return ((above + below) * weight).sum() / weight.sum().clamp(min=1e-8)


class CurveModel:
    """Predicts a dose-response curve per compound, enzyme and arm."""

    name = "curve"

    def __init__(
        self,
        targets: tuple[str, ...] = ISOFORMS,
        n_bits: int = 2048,
        surrogate: object | None = None,
        hidden: int = 256,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        n_forward_passes: int = 30,
        conditioning: str = "heads",
        descriptor_families: tuple[str, ...] = DESCRIPTOR_FAMILIES,
        seed: int = 0,
    ) -> None:
        self.conditioning = conditioning
        self.descriptor_families = descriptor_families
        self.targets = targets
        self.n_bits = n_bits
        #: Optional callable appended to the feature matrix, used to feed the
        #: single-concentration screen in as an input rather than as extra training
        #: rows. See :mod:`cypshift.data.surrogate`.
        self.surrogate = surrogate
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_forward_passes = n_forward_passes
        self.seed = seed
        self._model: CurveNetwork | None = None
        self._scaler: BlockScaler | None = None

    def _features(self, smiles: list[str], fit: bool) -> torch.Tensor:
        matrix = featurise(smiles, n_bits=self.n_bits, extra=self.surrogate).astype(np.float32)
        if fit:
            n_dense = len(physicochemical(["CCO"]).columns)
            self._scaler = BlockScaler(self.n_bits, n_dense).fit(matrix)
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
        arm_index = torch.tensor((usable["assay"] == "tdi_drc").to_numpy(int), dtype=torch.long)
        kind = torch.tensor(is_screen.astype(int), dtype=torch.long)
        lower_t = torch.tensor(lower, dtype=torch.float32)
        upper_t = torch.tensor(upper, dtype=torch.float32)
        weight_t = torch.tensor(usable["weight"].to_numpy(float), dtype=torch.float32)

        descriptors = None
        if self.conditioning == "descriptor":
            descriptors = torch.tensor(
                describe_all(list(self.targets), self.descriptor_families), dtype=torch.float32
            )
        self._model = CurveNetwork(
            features.shape[1],
            len(self.targets),
            len(ARMS),
            self.hidden,
            self.dropout,
            conditioning=self.conditioning,
            descriptors=descriptors,
        )
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        n = len(usable)
        logger.info(
            "training curves on %d observations, %d of them screen readings",
            n,
            int(is_screen.sum()),
        )

        for epoch in range(self.epochs):
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                predicted = self._model(features[molecule_index[batch]])
                rows = torch.arange(len(batch))
                parameters = predicted[rows, target_index[batch], arm_index[batch]]
                loss = curve_loss(
                    parameters, kind[batch], lower_t[batch], upper_t[batch], weight_t[batch]
                )
                loss.backward()
                optimiser.step()
                total += loss.detach().item() * len(batch)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d  loss %.4f", epoch, total / n)
        return self

    def _curves(self, smiles: list[str]) -> NDArray[np.float64]:
        """Curve parameters, shape ``(passes, compounds, targets, arms, 3)``."""
        if self._model is None:
            raise RuntimeError("model is not fitted")
        self._model.train()
        torch.manual_seed(self.seed)
        features = self._features(smiles, fit=False)
        with torch.no_grad():
            return np.stack([self._model(features).numpy() for _ in range(self.n_forward_passes)])

    def predict_pic50(self, smiles: list[str], arm: str = "direct") -> NDArray[np.float64]:
        """The curve's midpoint, which is what the regression track asks for."""
        return self._curves(smiles).mean(axis=0)[:, :, ARMS.index(arm), PIC50]

    def predict_response(
        self, smiles: list[str], pc: float = SCREEN_PC, arm: str = "direct"
    ) -> NDArray[np.float64]:
        """Inhibited fraction at any concentration, read off the predicted curve."""
        curves = torch.from_numpy(self._curves(smiles).mean(axis=0)[:, :, ARMS.index(arm)])
        return inhibition_at(curves, pc).numpy()

    def predict_is_time_dependent(
        self, smiles: list[str], assay_floor: float = 4.0
    ) -> NDArray[np.bool_]:
        """Compare the two arms' curves through the organisers' published rule."""
        curves = self._curves(smiles).mean(axis=0)
        direct = curves[:, :, ARMS.index("direct"), PIC50]
        preincubated = curves[:, :, ARMS.index("preincubated"), PIC50]
        return np.where(
            direct > assay_floor,
            (preincubated - direct) > TDI_SHIFT_THRESHOLD,
            preincubated > assay_floor + TDI_SHIFT_THRESHOLD,
        )
