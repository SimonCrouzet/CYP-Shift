"""Leave-one-target-out: the only test that can falsify a protein descriptor.

Conditioning a ligand model on a vector describing the protein is proteochemometrics,
and it has an evaluation problem that is easy to walk past. With eight enzymes the
model ever sees eight distinct conditioning vectors, so a descriptor of more than a
handful of dimensions is equivalent to a one-hot target index as far as the optimiser
is concerned: it memorises the lookup, fits well, and tells you nothing about whether
the *content* of the vector did any work.

Holding one enzyme out entirely is what separates the two. A one-hot cannot say
anything about an index it never trained on, so it must fall back to the population
mean. A descriptor can, if and only if it encodes something transferable -- that an
acidic pocket prefers basic ligands, say. The gap between them is the effect, and
without this comparison there is no evidence either way.

The comparison is against two baselines rather than one, because beating a one-hot is
necessary but not sufficient:

*population mean*
    Predict the training mean for the held-out enzyme. The floor.
*one-hot*
    An untrained index for the held-out enzyme, which is the population mean by another
    route but confirms the harness rather than assuming it.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cypshift.config import ISOFORMS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferResult:
    """What happened when one enzyme was held out of training entirely."""

    held_out: str
    n_train_observations: int
    n_held_out_observations: int
    score: float
    baseline_score: float

    @property
    def gain(self) -> float:
        """How much the descriptor bought over predicting the population mean."""
        return self.baseline_score - self.score


def split_by_target(observations: pd.DataFrame, held_out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition observations into everything-but-one-enzyme and that enzyme.

    Raises:
        ValueError: If the held-out enzyme has no observations, or if removing it
            would leave nothing to train on.
    """
    if held_out not in set(observations["target"]):
        raise ValueError(f"{held_out} has no observations to hold out")
    mask = observations["target"] == held_out
    train, test = observations.loc[~mask], observations.loc[mask]
    if train.empty:
        raise ValueError(f"holding out {held_out} leaves no training data")
    logger.info(
        "leave-one-target-out %s: train on %d observations, test on %d",
        held_out,
        len(train),
        len(test),
    )
    return train, test


def population_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Mean absolute error of predicting the training mean for the held-out enzyme.

    This is what a one-hot must fall back to, since it has never seen the held-out
    index. Any descriptor claiming to transfer has to beat it.
    """
    predicted = float(train["value"].mean())
    return float(np.abs(test["value"].to_numpy(float) - predicted).mean())


def leave_one_target_out(
    observations: pd.DataFrame,
    fit_predict: object,
    targets: tuple[str, ...] = ISOFORMS,
) -> pd.DataFrame:
    """Hold each enzyme out in turn and measure whether the model transfers.

    Args:
        observations: The full observation table.
        fit_predict: Callable taking ``(train, test)`` and returning predictions for
            ``test``, in its row order.
        targets: Enzymes to hold out, one at a time.

    Returns:
        One row per held-out enzyme, with the model's error, the population-mean
        baseline, and the gain between them. A gain at or below zero means the
        descriptor carried nothing across enzymes.
    """
    results = []
    for held_out in targets:
        train, test = split_by_target(observations, held_out)
        predicted = np.asarray(fit_predict(train, test), dtype=float)
        if len(predicted) != len(test):
            raise ValueError(f"expected {len(test)} predictions, got {len(predicted)}")
        error = float(np.abs(test["value"].to_numpy(float) - predicted).mean())
        baseline = population_mean_baseline(train, test)
        results.append(TransferResult(held_out, len(train), len(test), error, baseline))
        logger.info(
            "  %s: MAE %.3f against a population-mean baseline of %.3f (gain %+.3f)",
            held_out,
            error,
            baseline,
            baseline - error,
        )
    return pd.DataFrame(
        [
            {
                "held_out": r.held_out,
                "n_train": r.n_train_observations,
                "n_test": r.n_held_out_observations,
                "mae": r.score,
                "baseline_mae": r.baseline_score,
                "gain": r.gain,
            }
            for r in results
        ]
    ).set_index("held_out")
