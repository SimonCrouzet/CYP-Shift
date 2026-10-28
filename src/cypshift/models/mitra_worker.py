"""Fit Mitra regressors and predict with them, run as a script in the Mitra environment.

Mitra ships in AutoGluon, which pins pandas and pyarrow versions the main environment cannot
take, so it lives in its own environment (``/mnt/wsl/ext/envs/mitra``) and
:class:`cypshift.models.tfm.TabularFoundationModel` calls this file with that environment's
interpreter. This file imports nothing from ``cypshift`` and needs only numpy, torch and
AutoGluon.

Input ``--source`` is an ``.npz`` with ``x__<endpoint>`` / ``y__<endpoint>`` (the context rows of
each endpoint) and ``query`` (the rows to predict, shared by every endpoint). Output ``--target``
is an ``.npz`` with ``prediction__<endpoint>``. Mitra conditions on the context at predict time,
so one process fits and predicts every endpoint. With ``"task": "classification"`` in the settings
the labels are 0/1 and the prediction is the probability of 1.
"""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger("mitra_worker")


def mitra_regressor(settings: dict) -> object:
    """An AutoGluon ``MitraRegressor`` with the given ``fine_tune``, ``fine_tune_steps``,
    ``device`` and ``seed``."""
    from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor

    return MitraRegressor(
        device=settings["device"],
        fine_tune=settings["fine_tune"],
        fine_tune_steps=settings["fine_tune_steps"],
        seed=settings["seed"],
        verbose=False,
    )


def mitra_classifier(settings: dict) -> object:
    """An AutoGluon ``MitraClassifier`` with the given ``fine_tune``, ``fine_tune_steps``,
    ``device`` and ``seed``."""
    from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

    return MitraClassifier(
        device=settings["device"],
        fine_tune=settings["fine_tune"],
        fine_tune_steps=settings["fine_tune_steps"],
        seed=settings["seed"],
        verbose=False,
    )


def run(
    source: Path,
    target: Path,
    settings: dict,
    make_regressor: Callable[[dict], object] | None = None,
) -> None:
    """Fit one model per endpoint of ``source`` and write its predictions of ``query``: values for
    a regression task, the probability of the positive class for a classification task."""
    classify = settings.get("task") == "classification"
    if make_regressor is None:
        make_regressor = mitra_classifier if classify else mitra_regressor
    predictions = {}
    with np.load(source) as data:
        endpoints = sorted(key.split("__", 1)[1] for key in data.files if key.startswith("y__"))
        query = data["query"]
        for endpoint in endpoints:
            features, labels = data[f"x__{endpoint}"], data[f"y__{endpoint}"]
            # Mitra's validation split for fine-tuning draws from numpy's global generator.
            np.random.seed(settings["seed"])
            regressor = make_regressor(settings)
            if classify:
                regressor.fit(features, labels.astype(int))
                predicted = np.asarray(regressor.predict_proba(query), float)[:, 1]
            else:
                regressor.fit(features, labels)
                predicted = np.asarray(regressor.predict(query), float)
            predictions[f"prediction__{endpoint}"] = predicted
            logger.info("%s: fit on %d rows, predicted %d", endpoint, len(labels), len(query))
    np.savez(target, **predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--settings", required=True, help="JSON with fine_tune, fine_tune_steps, device, seed"
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [mitra] %(message)s")
    run(arguments.source, arguments.target, json.loads(arguments.settings))


if __name__ == "__main__":
    main()
