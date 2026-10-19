"""Passing the cheap screen in as a feature rather than as extra training rows.

There are 17,504 single-concentration screen readings against 6,325 dose-response
labels. Pooled into one training set they outnumber the labels we are scored on by
nearly three to one, and that is the failure mode Buterez's multi-fidelity benchmark
identifies: adding low-fidelity rows loses high-fidelity signal once the cheap data
dominates, and feeding a cheap model's *output* or its *hidden representation* in as an
input feature wins instead.

Both are built here.

``predicted_label``
    Fit a model on the screen, predict the inhibited fraction each compound would show
    at each enzyme, and hand those four numbers to the real model as extra columns. The
    high-fidelity model still trains only on dose-response labels, so the cheap data
    cannot outvote them -- it can only inform them.

``embedding``
    The same, but taking the screen model's hidden layer instead of its output. Wider,
    and it keeps structure the four output numbers discard, at the cost of dimensions
    that mean nothing individually.

**The screen_model is fitted on training compounds only.** Fitting it on everything and
then predicting for the holdout would let the screen readings of held-out compounds
reach the model through the feature, which inflates the holdout score without
improving anything on the blinded test set. :func:`fit_screen_model` takes the compounds
it may learn from explicitly for that reason.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor

from cypshift.config import ISOFORMS
from cypshift.data.screen import FOLD_CHANGE_COLUMN, inhibited_fraction, to_wide
from cypshift.features.molecular import featurise

logger = logging.getLogger(__name__)

SCREEN_MODEL_STRATEGIES = ("predicted_label", "embedding")

#: Width of the screen model's hidden layer, and therefore of the ``embedding``
#: screen_model. Small on purpose: it is an input to another model, and a wide block of
#: unnamed columns would swamp the fingerprint it sits beside.
DEFAULT_EMBEDDING_WIDTH = 16

#: Below this many training compounds, early stopping is switched off: its 10%
#: validation split would be empty and scikit-learn raises rather than degrading.
MIN_COMPOUNDS_FOR_EARLY_STOPPING = 30


@dataclass
class ScreenModel:
    """A model of the screen, used to featurise compounds for the real model."""

    network: MLPRegressor
    strategy: str
    n_bits: int
    columns: tuple[str, ...]
    #: Centre and spread of the screen_model's own output on the training compounds.
    #: The screen_model standardises itself because of where its columns land: they are
    #: appended last, and BlockScaler treats trailing columns as binary and passes them
    #: through unscaled. An embedding block averaging 6.2 would then sit unscaled
    #: beside a sparse fingerprint -- the same mistake that cost 0.089 when uniform
    #: scaling was applied across blocks of different kinds.
    centre: NDArray[np.float32] | None = None
    spread: NDArray[np.float32] | None = None

    def _raw(self, smiles: list[str]) -> NDArray[np.float32]:
        x = featurise(smiles, n_bits=self.n_bits)
        if self.strategy == "predicted_label":
            return np.asarray(self.network.predict(x), dtype=np.float32).reshape(len(smiles), -1)
        return hidden_activations(self.network, x)

    def transform(self, smiles: list[str]) -> NDArray[np.float32]:
        """Standardised screen_model features for a list of compounds."""
        values = self._raw(list(smiles))
        if self.centre is None or self.spread is None:
            return values
        return ((values - self.centre) / self.spread).astype(np.float32)

    def __call__(self, smiles: list[str]) -> NDArray[np.float32]:
        """So a ScreenModel can be handed straight to ``featurise(extra=...)``."""
        return self.transform(list(smiles))


def hidden_activations(network: MLPRegressor, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Activations of the last hidden layer.

    scikit-learn exposes no hook for this, so the forward pass is repeated from the
    fitted weights. ReLU throughout, matching ``activation="relu"``; the output layer
    is deliberately not applied, since its output is what ``predicted_label`` already
    provides.
    """
    activation = np.asarray(x, dtype=np.float64)
    for weight, bias in zip(network.coefs_[:-1], network.intercepts_[:-1], strict=True):
        activation = np.maximum(activation @ weight + bias, 0.0)
    return activation.astype(np.float32)


def screen_targets(screen: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Inhibited fraction per compound and enzyme, one column each.

    Returns the wide table and the enzyme column names present, which need not be all
    four -- a screen subset may not cover every enzyme.
    """
    wide = to_wide(screen, FOLD_CHANGE_COLUMN)
    enzymes = [c for c in wide.columns if c not in ("Molecule_Name", "SMILES")]
    for enzyme in enzymes:
        wide[enzyme] = inhibited_fraction(wide[enzyme].to_numpy(dtype=float))
    return wide, enzymes


def fit_screen_model(
    screen: pd.DataFrame,
    trainable: set[str],
    strategy: str,
    n_bits: int = 2048,
    hidden: int = DEFAULT_EMBEDDING_WIDTH,
    seed: int = 0,
) -> ScreenModel:
    """Fit a model of the screen on the compounds it is allowed to see.

    Args:
        screen: The long single-concentration table.
        trainable: Molecule names the screen_model may learn from. Compounds outside this
            set are dropped even if the screen covers them, so a holdout compound's
            screen reading cannot reach the model through the feature.
        strategy: One of :data:`SCREEN_MODEL_STRATEGIES`.
        n_bits: Fingerprint width, matched to the downstream model's.
        hidden: Width of the hidden layer, and of an ``embedding`` screen_model.
        seed: Passed to the network.

    Raises:
        ValueError: On an unknown strategy, or if no trainable compound has a reading.
    """
    if strategy not in SCREEN_MODEL_STRATEGIES:
        raise ValueError(f"unknown screen_model strategy {strategy!r}")

    wide, enzymes = screen_targets(screen)
    wide = wide[wide["Molecule_Name"].isin(trainable)]
    if wide.empty:
        raise ValueError("no trainable compound has a screen reading")
    wide = wide.dropna(subset=enzymes)
    if wide.empty:
        raise ValueError("no trainable compound has a complete screen row")

    x = featurise(wide["SMILES"].tolist(), n_bits=n_bits)
    y = wide[enzymes].to_numpy(dtype=float)
    # Early stopping holds back 10% as a validation split, which sklearn refuses when
    # that slice would be empty. Small training sets are not hypothetical here: fitting
    # on retained compounds only, or on a screen subset covering few compounds, gets
    # there easily. Below the threshold the network simply trains to max_iter.
    early_stopping = len(wide) >= MIN_COMPOUNDS_FOR_EARLY_STOPPING
    if not early_stopping:
        logger.info("only %d compounds; training without early stopping", len(wide))
    network = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        random_state=seed,
        max_iter=400,
        early_stopping=early_stopping,
    )
    network.fit(x, y)
    logger.info(
        "fitted %s screen_model on %d compounds over %d enzymes",
        strategy,
        len(wide),
        len(enzymes),
    )
    columns = (
        tuple(f"screen_predicted_{e}" for e in enzymes)
        if strategy == "predicted_label"
        else tuple(f"screen_embedding_{i}" for i in range(hidden))
    )
    screen_model = ScreenModel(network, strategy, n_bits, columns)
    # Standardise on the training compounds only, for the same reason the network is
    # fitted on them only.
    raw = screen_model._raw(wide["SMILES"].tolist())
    screen_model.centre = raw.mean(axis=0)
    screen_model.spread = raw.std(axis=0) + 1e-6
    return screen_model


def screen_model_frame(screen_model: ScreenModel, smiles: list[str]) -> pd.DataFrame:
    """ScreenModel features as a named table, for joining onto an observation set."""
    values = screen_model.transform(smiles)
    return pd.DataFrame(
        values, columns=list(screen_model.columns), index=pd.Index(smiles, name="SMILES")
    )


def isoform_columns() -> tuple[str, ...]:
    """Enzyme names as the screen table spells them."""
    return tuple(name.split("_")[0] for name in ISOFORMS)
