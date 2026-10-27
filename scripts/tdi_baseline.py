"""Direct is_TDI classifier baseline: LightGBM on molecular features, scored by cross-validated MCC.

    python scripts/tdi_baseline.py --output scratch/tdi_baseline_oof.csv

The models in the repo call ``is_TDI`` from the gap between two predicted potencies. This trains a
classifier on the ``is_TDI`` labels directly, one per enzyme (CYP2D6, CYP3A4), on the same
fingerprint, physicochemical and mechanistic features as the boosted regression model, in
scaffold-grouped folds of the TDI training table. It prints, per enzyme and as the macro mean the
leaderboard ranks by, the MCC at a fixed 0.5 probability, at the threshold that predicts as many
positives as the training prevalence (no fitting on the held-out scores), and at the best threshold
on the out-of-fold scores (optimistic), with ROC AUC and average precision for the ranking alone.
"""

import argparse
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.data.splits import stratified_scaffold_folds  # noqa: E402 -- after the src path
from cypshift.features.molecular import featurise  # noqa: E402

logger = logging.getLogger("tdi_baseline")

TDI_TABLE = Path("data/raw/cyp-challenge-TRAIN_TDI.csv")
ENZYMES = ("CYP2D6", "CYP3A4")


def mcc_at(truth: np.ndarray, score: np.ndarray, threshold: float) -> float:
    """MCC of calling a compound positive when its score is at least ``threshold``."""
    return float(matthews_corrcoef(truth, score >= threshold))


def mcc_at_prevalence(truth: np.ndarray, score: np.ndarray, prevalence: float) -> float:
    """MCC when the top ``prevalence`` share of the scores is called positive."""
    cut = np.quantile(score, 1.0 - prevalence)
    return mcc_at(truth, score, cut)


def best_threshold_mcc(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """The threshold on ``score`` with the highest MCC, and that MCC (fitted on these scores)."""
    best = (0.5, -1.0)
    for threshold in np.unique(np.quantile(score, np.linspace(0.5, 0.99, 99))):
        value = mcc_at(truth, score, threshold)
        if value > best[1]:
            best = (float(threshold), value)
    return best


def predicted_potency(table: pd.DataFrame, paths: list[Path]) -> np.ndarray:
    """The mean over ``paths`` of the predicted direct pIC50 of the four enzymes, for the compounds
    of ``table``: shape ``(len(table), 4)``, NaN for a compound the predictions do not cover."""
    columns = [
        f"{e}_pIC50_direct_inhibition_pred" for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
    ]
    frames = [pd.read_parquet(path).set_index("Molecule_Name")[columns] for path in paths]
    mean = sum(frames) / len(frames)
    return mean.reindex(table["Molecule_Name"]).to_numpy(dtype=np.float32)


def out_of_fold_scores(
    features: np.ndarray, labels: np.ndarray, folds: np.ndarray, threads: int, seed: int
) -> np.ndarray:
    """Out-of-fold positive-class probabilities of a LightGBM classifier; NaN where unlabelled."""
    scores = np.full(len(labels), np.nan)
    labelled = ~np.isnan(labels)
    for fold in np.unique(folds):
        train = labelled & (folds != fold)
        test = labelled & (folds == fold)
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.3,
            reg_lambda=1.0,
            n_jobs=threads,
            random_state=seed,
            verbose=-1,
        )
        model.fit(features[train], labels[train].astype(int))
        scores[test] = model.predict_proba(features[test])[:, 1]
        logger.info("fold %d: %d train, %d held out", fold + 1, train.sum(), test.sum())
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--table", type=Path, default=TDI_TABLE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bits", type=int, default=4096)
    parser.add_argument(
        "--direct-potency",
        action="store_true",
        help="add the enzyme's measured direct pIC50 as a feature (an upper bound for stacking "
        "a predicted potency: the blinded test has none)",
    )
    parser.add_argument(
        "--predicted-potency",
        type=Path,
        nargs="+",
        help="out-of-fold regression predictions (out_of_fold.parquet of run bundles): their mean "
        "predicted direct pIC50 of the four enzymes is added as features (NaN where a compound "
        "has none)",
    )
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    smiles = table["SMILES"].tolist()
    logger.info("featurising %d compounds", len(smiles))
    features = featurise(smiles, fingerprint=True, physchem=True, mechanism=True, n_bits=args.bits)
    if args.predicted_potency:
        features = np.hstack([features, predicted_potency(table, args.predicted_potency)])
    labels = {e: table[f"{e}_is_TDI"].astype(float) for e in ENZYMES}
    strata = sum(
        (labels[e].fillna(0).to_numpy() * (2**i)).astype(int) for i, e in enumerate(ENZYMES)
    )
    folds = stratified_scaffold_folds(smiles, strata, args.folds, args.seed)

    out = table[["Molecule_Name", "SMILES"]].copy()
    out["fold"] = folds
    rows = []
    for enzyme in ENZYMES:
        truth = labels[enzyme].to_numpy()
        columns = features
        if args.direct_potency:
            potency = table[f"{enzyme}_pIC50_direct_inhibition"].to_numpy(dtype=np.float32)
            columns = np.hstack([features, potency[:, None]])
        scores = out_of_fold_scores(columns, truth, folds, args.threads, args.seed)
        out[f"{enzyme}_is_TDI"] = truth
        out[f"{enzyme}_score"] = scores
        keep = ~np.isnan(truth)
        y, s = truth[keep].astype(int), scores[keep]
        prevalence = float(y.mean())
        threshold, best = best_threshold_mcc(y, s)
        rows.append(
            {
                "enzyme": enzyme,
                "n": int(keep.sum()),
                "prevalence": prevalence,
                "MCC@0.5": mcc_at(y, s, 0.5),
                "MCC@prevalence": mcc_at_prevalence(y, s, prevalence),
                "MCC@best": best,
                "best_threshold": threshold,
                "ROC_AUC": float(roc_auc_score(y, s)),
                "AP": float(average_precision_score(y, s)),
            }
        )
    report = pd.DataFrame(rows)
    macro = report[["MCC@0.5", "MCC@prevalence", "MCC@best", "ROC_AUC", "AP"]].mean()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    logger.info("\n%s", report.round(4).to_string(index=False))
    logger.info("macro mean over the two enzymes:\n%s", macro.round(4).to_string())
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
