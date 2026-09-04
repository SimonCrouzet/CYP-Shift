"""Hold one enzyme out of training entirely and see whether the descriptor transfers.

The question this answers cannot be answered by the holdout score. With four scored
enzymes the model ever sees four distinct conditioning vectors, so a descriptor of any
width can be memorised as a lookup, fit well, and reveal nothing about whether its
contents mean anything. Holding an enzyme out breaks the lookup: a per-enzyme head has
never trained the held-out one and can only fall back to the population mean, while a
descriptor can do better if and only if it encodes something transferable.

The comparison worth reading is between conditioning modes at a fixed descriptor, and
between descriptors at a fixed mode. An absolute number here is not comparable to a
leaderboard score -- the model is predicting an enzyme it has never seen.

Usage::

    python scripts/run_transfer_test.py
    python scripts/run_transfer_test.py --conditioning film --sources curated srs
"""

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import ISOFORMS
from cypshift.data.assembly import build_observations
from cypshift.data.challenge import load
from cypshift.data.conditioning import DESCRIPTOR_SOURCES
from cypshift.data.splits import similarity_matched_holdout
from cypshift.eval.transfer import leave_one_target_out
from cypshift.models.curve import CONDITIONING, CurveModel

logger = logging.getLogger(__name__)


def build_fit_predict(
    conditioning: str, sources: tuple[str, ...], epochs: int, seed: int
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """A ``(train, test) -> predictions`` callable for :func:`leave_one_target_out`.

    The model is rebuilt per fold, since a model that saw the held-out enzyme in an
    earlier fold would carry it forward and quietly answer a different question.
    """

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        targets = tuple(sorted(set(train["target"]) | set(test["target"])))
        model = CurveModel(
            targets=targets,
            conditioning=conditioning,
            descriptor_sources=sources,
            epochs=epochs,
            seed=seed,
        )
        model.fit(train)
        held_out = test["target"].iloc[0]
        column = targets.index(held_out)
        predicted = model.predict_pic50(test["SMILES"].tolist())
        return predicted[:, column]

    return fit_predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--conditioning", choices=CONDITIONING, default="film")
    parser.add_argument("--sources", nargs="+", choices=DESCRIPTOR_SOURCES, default=["curated"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.conditioning == "heads":
        logger.info(
            "conditioning=heads has no descriptor: a held-out enzyme's head is "
            "untrained, so this measures the floor the descriptors must beat"
        )

    train = load("inhibition", args.directory)
    mask = similarity_matched_holdout(
        train["SMILES"].tolist(),
        load("test", args.directory)["SMILES"].tolist(),
        seed=args.seed,
    )
    observations = build_observations(train.loc[~mask])
    observations = observations[observations["assay"] == "direct_inhibition_drc"]

    scored = tuple(name.split("_")[0] for name in ISOFORMS)
    results = leave_one_target_out(
        observations,
        build_fit_predict(args.conditioning, tuple(args.sources), args.epochs, args.seed),
        targets=scored,
    )
    results["conditioning"] = args.conditioning
    results["sources"] = ",".join(args.sources)

    pd.set_option("display.width", 200)
    print(results.to_string())
    print(f"\nmean gain over the population mean: {results['gain'].mean():+.4f}")
    print("a gain at or below zero means the descriptor carried nothing across enzymes")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output)
        logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
