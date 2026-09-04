"""One model of the latent potencies, from which every requested output derives.

The three data sources are not three tasks. A twelve-point curve and a
single-concentration screen are two *measurements of the same quantity* through
different instruments -- the curve resolves an IC50 with a credible interval, the
screen sees a saturating, censored, one-point readout of it. That difference belongs
in the likelihood, not in the input, so the measurement channel is not conditioned on
at all.

Preincubation is different in kind. It changes what is physically being measured:
enzyme destroyed by turnover rather than enzyme occupied. So it is a real condition on
the quantity, and the model predicts two latent numbers per compound and enzyme:

    direct   -- pIC50 without preincubation
    shift    -- the leftward displacement after preincubation, in log units

Everything the challenge asks for is a function of those two. The regression track
wants ``direct``. The classification track wants the published rule applied to
``direct`` and ``direct + shift``. Nothing else has to be trained.

Predicting the shift rather than the preincubated potency directly is deliberate. The
shift is small -- median 0.15 to 0.31 log units -- and strongly correlated with the
direct value, so a model that predicts both potencies independently spends its
capacity re-learning the same molecule twice and lets the difference between two large
noisy numbers carry the signal. Predicting the difference makes it the thing being
fitted.
"""

import logging

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from cypshift.config import ISOFORMS, TDI_SHIFT_THRESHOLD
from cypshift.features.molecular import featurise, physicochemical
from cypshift.features.scaling import BlockScaler

logger = logging.getLogger(__name__)

DIRECT, SHIFT = 0, 1
UNBOUNDED = 1e4

#: Which latent quantity an observation constrains. A direct-arm measurement pins
#: ``direct``; a preincubated one pins ``direct + shift``; a screen reading is a
#: censored view of ``direct``.
ARM_DIRECT, ARM_PREINCUBATED = "direct", "preincubated"


class LatentPotencyNetwork(nn.Module):
    """Shared trunk over molecular features, two outputs per enzyme."""

    def __init__(
        self, n_features: int, n_targets: int, hidden: int = 256, dropout: float = 0.2
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
        # One head per enzyme, each emitting (direct, shift). Separate heads rather
        # than a conditioning vector, because the enzymes share chemistry in the trunk
        # but not their recognition: CYP2C9 and CYP3A4 correlate at 0.69 while CYP2D6
        # correlates with everything at 0.06.
        self.heads = nn.ModuleList([nn.Linear(hidden // 2, 2) for _ in range(n_targets)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``(batch, n_targets, 2)``."""
        shared = self.trunk(x)
        return torch.stack([head(shared) for head in self.heads], dim=1)


def observation_loss(
    direct: torch.Tensor,
    shift: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    is_preincubated: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Soft-threshold error against whichever latent quantity a row observes.

    A direct-arm row is compared with ``direct``; a preincubated row with
    ``direct + shift``. The screen enters through its own wide, one-sided interval
    rather than through a separate term, since it measures the same quantity.
    """
    predicted = direct + shift * is_preincubated
    above = torch.clamp(predicted - upper, min=0.0)
    below = torch.clamp(lower - predicted, min=0.0)
    return ((above + below) * weight).sum() / weight.sum().clamp(min=1e-8)


class LatentPotencyModel:
    """Joint model of direct potency and the preincubation shift."""

    name = "latent_potency"

    def __init__(
        self,
        targets: tuple[str, ...] = ISOFORMS,
        n_bits: int = 2048,
        hidden: int = 256,
        dropout: float = 0.2,
        epochs: int = 60,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        shift_penalty: float = 0.01,
        n_forward_passes: int = 30,
        seed: int = 0,
    ) -> None:
        self.targets = targets
        self.n_bits = n_bits
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.shift_penalty = shift_penalty
        self.n_forward_passes = n_forward_passes
        self.seed = seed
        self._model: LatentPotencyNetwork | None = None
        self._scaler: BlockScaler | None = None
        self._centre = 0.0
        self._spread = 1.0

    def _features(self, smiles: list[str], fit: bool) -> torch.Tensor:
        matrix = featurise(smiles, n_bits=self.n_bits).astype(np.float32)
        if fit:
            n_dense = len(physicochemical(["CCO"]).columns)
            self._scaler = BlockScaler(n_count_columns=self.n_bits, n_dense_columns=n_dense).fit(
                matrix
            )
        if self._scaler is None:
            raise RuntimeError("model is not fitted")
        return torch.from_numpy(self._scaler.transform(matrix))

    def fit(self, observations: pd.DataFrame) -> "LatentPotencyModel":
        """Train on the observation table, using each row's arm and interval.

        Args:
            observations: Rows carrying ``SMILES``, ``target``, ``assay``, ``value``,
                ``conf_low``, ``conf_high``, ``weight`` and ``is_censored``.
        """
        torch.manual_seed(self.seed)
        usable = observations[np.isfinite(observations["value"])].reset_index(drop=True)
        if usable.empty:
            raise ValueError("no finite measurements to fit on")

        molecules = usable["SMILES"].drop_duplicates().tolist()
        row_of = {smiles: index for index, smiles in enumerate(molecules)}
        features = self._features(molecules, fit=True)

        value = usable["value"].to_numpy(float)
        self._centre, self._spread = float(value.mean()), float(value.std()) or 1.0
        lower = np.where(np.isfinite(usable["conf_low"]), usable["conf_low"], value)
        upper = np.where(np.isfinite(usable["conf_high"]), usable["conf_high"], value)
        lower = np.where(usable["is_censored"].to_numpy(bool), -UNBOUNDED, lower)

        molecule_index = torch.tensor([row_of[s] for s in usable["SMILES"]], dtype=torch.long)
        target_index = torch.tensor(
            [self.targets.index(t) for t in usable["target"]], dtype=torch.long
        )
        preincubated = torch.tensor(
            (usable["assay"] == "tdi_drc").to_numpy(float), dtype=torch.float32
        )
        lower_t = torch.tensor((lower - self._centre) / self._spread, dtype=torch.float32)
        upper_t = torch.tensor((upper - self._centre) / self._spread, dtype=torch.float32)
        weight_t = torch.tensor(usable["weight"].to_numpy(float), dtype=torch.float32)

        self._model = LatentPotencyNetwork(
            features.shape[1], len(self.targets), self.hidden, self.dropout
        )
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        n = len(usable)
        logger.info(
            "training on %d observations over %d molecules, %d preincubated",
            n,
            len(molecules),
            int(preincubated.sum()),
        )

        for epoch in range(self.epochs):
            order = torch.randperm(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimiser.zero_grad()
                latent = self._model(features[molecule_index[batch]])
                rows = torch.arange(len(batch))
                direct = latent[rows, target_index[batch], DIRECT]
                shift = latent[rows, target_index[batch], SHIFT]
                loss = observation_loss(
                    direct,
                    shift,
                    lower_t[batch],
                    upper_t[batch],
                    preincubated[batch],
                    weight_t[batch],
                )
                # The shift is small and mostly zero; without a penalty the model can
                # absorb direct-arm error into it wherever a compound has both arms.
                loss = loss + self.shift_penalty * shift.abs().mean()
                loss.backward()
                optimiser.step()
                total += loss.detach().item() * len(batch)
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d  loss %.4f", epoch, total / n)
        return self

    def _latent(self, smiles: list[str]) -> NDArray[np.float64]:
        """Monte Carlo latent estimates, shape ``(passes, compounds, targets, 2)``.

        Dropout is left on as the uncertainty source, so the generator is seeded first.
        That makes predictions reproducible, and it makes the derived outputs mutually
        consistent: ``predict_preincubated`` must equal ``predict_direct`` plus
        ``predict_shift``, which fails if each call draws its own dropout masks.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted")
        self._model.train()
        torch.manual_seed(self.seed)
        features = self._features(smiles, fit=False)
        with torch.no_grad():
            draws = np.stack([self._model(features).numpy() for _ in range(self.n_forward_passes)])
        draws[..., DIRECT] = draws[..., DIRECT] * self._spread + self._centre
        draws[..., SHIFT] = draws[..., SHIFT] * self._spread
        return draws

    def predict_direct(self, smiles: list[str]) -> NDArray[np.float64]:
        """Direct-inhibition pIC50 per compound and enzyme. The regression track."""
        return self._latent(smiles)[..., DIRECT].mean(axis=0)

    def predict_shift(self, smiles: list[str]) -> NDArray[np.float64]:
        """Preincubation shift in log units, per compound and enzyme."""
        return self._latent(smiles)[..., SHIFT].mean(axis=0)

    def predict_preincubated(self, smiles: list[str]) -> NDArray[np.float64]:
        """pIC50 after preincubation, which is direct plus the shift."""
        latent = self._latent(smiles).mean(axis=0)
        return latent[..., DIRECT] + latent[..., SHIFT]

    def predict_is_time_dependent(
        self, smiles: list[str], assay_floor: float = 4.0
    ) -> NDArray[np.bool_]:
        """Apply the challenge's own labelling rule to the predicted latents.

        The rule is reproduced exactly from the published FAQ: above the assay floor a
        compound is positive when the shift exceeds two-fold; below it, positive when
        the preincubated value clears the floor by that same margin.
        """
        latent = self._latent(smiles).mean(axis=0)
        direct = latent[..., DIRECT]
        preincubated = direct + latent[..., SHIFT]
        resolved = direct > assay_floor
        return np.where(
            resolved,
            (preincubated - direct) > TDI_SHIFT_THRESHOLD,
            preincubated > assay_floor + TDI_SHIFT_THRESHOLD,
        )
