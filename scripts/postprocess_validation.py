#!/usr/bin/env python
"""Score the base model and each postprocessor on one outer holdout, fit on the inner side.

The calibration script reports before/after error on the same out-of-fold rows it fits the
map on, and the analogue corrector's reference pool, if drawn from the full out-of-fold table,
contains the compounds being scored. Neither is independent evidence that a postprocessor
helps. This runs the whole chain under one outer split instead:

  1. Split the training compounds into an outer holdout and an inner set.
  2. Inner out-of-fold: k-fold CV over the inner set alone, giving one prediction per inner
     compound and the residuals the corrector's reference pool is built from.
  3. Base: refit the configured model on the whole inner set, predict the outer holdout.
  4. Calibrate: fit the affine map on the inner out-of-fold, apply to the base outer predictions.
  5. Correct: fit the analogue corrector's reference pool on the inner set (truth + inner
     out-of-fold prediction), apply to the base outer predictions.
  6. Score base / calibrated / corrected on the identical outer-holdout molecules, observed
     labels only. The outer holdout's own labels never enter the map, the reference pool, or
     the inner fit.

    python scripts/postprocess_validation.py --set model.kind=graph \\
        --outer analogue --calibration-method affine --estimator residual
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cypshift.config import ISOFORMS, REGRESSION_ENDPOINTS
from cypshift.data.challenge import load
from cypshift.data.splits import (
    analogue_split,
    similarity_matched_holdout,
    stratified_scaffold_folds,
)
from cypshift.decision.analogue import ESTIMATORS, AnalogueCorrector
from cypshift.eval.scoring import score_submission
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import assemble_training_data, build_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import METHODS as CALIBRATION_METHODS  # noqa: E402
from calibrate import affine_for_endpoint  # noqa: E402


def _targets(observations: pd.DataFrame) -> tuple[str, ...]:
    scored = list(ISOFORMS)
    extra = [t for t in dict.fromkeys(observations["target"]) if t not in scored]
    return tuple(scored + sorted(extra))


def fit_and_predict(
    config: ExperimentConfig,
    retained: pd.DataFrame,
    query_smiles: list[str],
    exclude_names: set[str],
) -> np.ndarray:
    """Fit the configured model on ``retained`` and return its mean prediction for the queries."""
    observations, screen_model = assemble_training_data(
        config, retained, exclude_names=exclude_names
    )
    model = build_model(config, screen_model, targets=_targets(observations))
    model.fit(observations)
    return model.predict(query_smiles).mean


def inner_out_of_fold(
    config: ExperimentConfig,
    inner: pd.DataFrame,
    test_names: set[str],
    outer_names: set[str],
    n_folds: int,
) -> np.ndarray:
    """One out-of-fold prediction per inner compound, from a k-fold CV over the inner set alone.

    Returns an ``(n_inner, n_endpoints)`` array aligned to ``inner``'s row order. The test and
    outer-holdout compounds are kept out of every fold's TDI training path.
    """
    strata = inner[list(REGRESSION_ENDPOINTS)].notna().sum(axis=1).to_numpy()
    folds = stratified_scaffold_folds(
        inner["SMILES"].tolist(), strata, n_folds=n_folds, seed=config.data.seed
    )
    predictions = np.full((len(inner), len(REGRESSION_ENDPOINTS)), np.nan)
    for fold in range(n_folds):
        held = folds == fold
        retained = inner.loc[~held]
        evaluation = inner.loc[held]
        exclude = set(evaluation["Molecule_Name"]) | test_names | outer_names
        predictions[np.flatnonzero(held)] = fit_and_predict(
            config, retained, evaluation["SMILES"].tolist(), exclude
        )
    return predictions


def _truth_block(frame: pd.DataFrame, suffix: str) -> np.ndarray:
    cols = [f"{e}{suffix}" if suffix else e for e in REGRESSION_ENDPOINTS]
    return np.column_stack([frame[c].to_numpy(float) for c in cols])


def _submission(names: pd.Series, values: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"Molecule_Name": names.to_numpy()})
    for column, endpoint in enumerate(REGRESSION_ENDPOINTS):
        out[endpoint] = values[:, column]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", metavar="section.field=value", default=[])
    parser.add_argument(
        "--outer", choices=("analogue", "scaffold", "similarity"), default="analogue"
    )
    parser.add_argument("--outer-frac", type=float, default=0.15)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--calibration-method", choices=CALIBRATION_METHODS, default="affine")
    parser.add_argument("--estimator", choices=ESTIMATORS, default="residual")
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    arguments = parser.parse_args(argv)

    config = (
        ExperimentConfig.load(Path(arguments.config))
        if arguments.config
        else ExperimentConfig.load()
    )
    for assignment in arguments.set:
        dotted, _, raw = assignment.partition("=")
        try:
            value = json.loads(raw.strip())
        except json.JSONDecodeError:
            value = raw.strip()
        config.override(dotted.strip(), value)
    config.validate()

    train = load("inhibition", arguments.directory)
    test = load("test", arguments.directory)
    test_names = set(test["Molecule_Name"])

    if arguments.outer == "analogue":
        split = analogue_split(
            train["SMILES"].tolist(), holdout_fraction=arguments.outer_frac, seed=config.data.seed
        )
        outer = split.holdout_mask
        coverage = split.anchor_label_coverage(train[list(REGRESSION_ENDPOINTS)].notna().to_numpy())
        print(
            "same-endpoint anchor coverage:",
            {
                REGRESSION_ENDPOINTS[c].split("_")[0]: (None if v != v else round(v, 3))
                for c, v in coverage.items()
            },
        )
    elif arguments.outer == "similarity":
        outer = similarity_matched_holdout(
            train["SMILES"].tolist(),
            test["SMILES"].tolist(),
            holdout_fraction=arguments.outer_frac,
            seed=config.data.seed,
        )
    else:
        strata = train[list(REGRESSION_ENDPOINTS)].notna().sum(axis=1).to_numpy()
        n_folds = max(2, round(1 / arguments.outer_frac))
        outer = (
            stratified_scaffold_folds(
                train["SMILES"].tolist(), strata, n_folds=n_folds, seed=config.data.seed
            )
            == 0
        )

    inner = train.loc[~outer].reset_index(drop=True)
    outer_frame = train.loc[outer].reset_index(drop=True)
    outer_names = set(outer_frame["Molecule_Name"])
    print(f"outer holdout: {len(outer_frame)} compounds ({arguments.outer}); inner: {len(inner)}")

    # Inner out-of-fold, then the base model refit on the whole inner set and predicting outer.
    inner_oof = inner_out_of_fold(config, inner, test_names, outer_names, arguments.inner_folds)
    base = fit_and_predict(config, inner, outer_frame["SMILES"].tolist(), outer_names | test_names)

    # Calibration map fit on the inner out-of-fold only.
    inner_truth = _truth_block(inner, "")
    inner_low = _truth_block(inner, "_conf_low")
    inner_high = _truth_block(inner, "_conf_high")
    calibrated = base.copy()
    for column in range(len(REGRESSION_ENDPOINTS)):
        finite = np.isfinite(inner_truth[:, column]) & np.isfinite(inner_oof[:, column])
        a, b = affine_for_endpoint(
            inner_truth[finite, column],
            inner_oof[finite, column],
            inner_low[finite, column],
            inner_high[finite, column],
            arguments.calibration_method,
            0.5,
        )
        calibrated[:, column] = a * base[:, column] + b

    # Analogue corrector reference pool fit on the inner set (its labels and inner-OOF residuals).
    corrector = AnalogueCorrector(estimator=arguments.estimator).fit(
        inner["SMILES"].tolist(), inner_truth, inner_low, inner_high, inner_oof
    )
    corrected = corrector.correct(outer_frame["SMILES"].tolist(), base)

    print(f"\n=== outer-holdout macro ST-RAE ({config.model.kind}) ===")
    for name, values in (("base", base), ("calibrated", calibrated), ("corrected", corrected)):
        scores = score_submission(_submission(outer_frame["Molecule_Name"], values), outer_frame)
        macro = scores.loc["MA", "ST-RAE_mean"]
        per = {
            e.split("_")[0]: round(scores.loc[e, "ST-RAE_mean"], 4) for e in REGRESSION_ENDPOINTS
        }
        print(f"{name:>10}: macro {macro:.4f} | {per}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
