"""Direct is_TDI classifier baseline: LightGBM, TabPFN or Mitra on molecular features, scored by
cross-validated MCC.

    python scripts/tdi_baseline.py --output scratch/tdi_baseline_oof.csv
    python scripts/tdi_baseline.py --model tabpfn --output scratch/tdi_tabpfn_oof.csv

``--model tabpfn`` and ``--model mitra`` are the tabular foundation models. ``--features`` picks
one of the regression comparison's feature sets (dense: physicochemical + mechanistic; chemeleon:
the CheMeleon embedding; dense_chemeleon; fp_dense: Morgan fingerprint + dense). The foundation
models read them as the regression ones do: impute, standardise, and PCA to 500 components when a
set is wider than that, fitted on the training fold; LightGBM reads them raw. Mitra runs in its own
environment (``MITRA_PYTHON``).

The models in the repo call ``is_TDI`` from the gap between two predicted potencies. This trains a
classifier on the ``is_TDI`` labels directly, one per enzyme (CYP2D6, CYP3A4), on the same
fingerprint, physicochemical and mechanistic features as the boosted regression model, in
scaffold-grouped folds of the TDI training table. It prints, per enzyme and as the macro mean the
leaderboard ranks by, the MCC at a fixed 0.5 probability, at the threshold that predicts as many
positives as the training prevalence (no fitting on the held-out scores), and at the best threshold
on the out-of-fold scores (optimistic), with ROC AUC and average precision for the ranking alone.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.data.splits import stratified_scaffold_folds  # noqa: E402 -- after the src path
from cypshift.features.assembly import molecular_feature_matrix  # noqa: E402

logger = logging.getLogger("tdi_baseline")
MODELS = ("lightgbm", "tabpfn", "mitra")
#: The feature sets of the regression comparison: (fingerprint, physchem + mechanistic, embedding).
FEATURE_SETS = {
    "dense": (False, True, "none"),
    "chemeleon": (False, False, "chemeleon"),
    "dense_chemeleon": (False, True, "chemeleon"),
    "fp_dense": (True, True, "none"),
}
PCA_COMPONENTS = 500
MITRA_PYTHON = "/mnt/wsl/ext/envs/mitra/bin/python"
MITRA_WORKER = Path(__file__).resolve().parents[1] / "src/cypshift/models/mitra_worker.py"

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


def bundle_features(
    table: pd.DataFrame, bundle: Path, embedding: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The stacked columns of a graph run's bundle for the compounds of ``table`` and which of them
    the run's folds covered: ``(features, covered)``.

    ``embedding`` reads ``out_of_fold_embedding.parquet`` (the encoder's encodings); otherwise the
    columns are every head of the run (``out_of_fold_auxiliary.parquet``), its direct pIC50
    predictions, and per enzyme the preincubated head minus the direct one where the run has both.
    Each compound's values are from the fold model that did not train on it.
    """
    names = table["Molecule_Name"]
    if embedding:
        frame = pd.read_parquet(bundle / "out_of_fold_embedding.parquet").set_index("Molecule_Name")
        return frame.reindex(names).to_numpy(np.float32), names.isin(frame.index).to_numpy()
    direct = pd.read_parquet(bundle / "out_of_fold.parquet").set_index("Molecule_Name")
    direct = direct[[c for c in direct.columns if c.endswith("_pred")]]
    heads = pd.read_parquet(bundle / "out_of_fold_auxiliary.parquet").set_index("Molecule_Name")
    frame = direct.join(heads)
    for enzyme in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
        endpoint = f"{enzyme}_pIC50_direct_inhibition"
        if f"{endpoint}__tdi_drc" in frame:
            frame[f"{endpoint}__shift"] = frame[f"{endpoint}__tdi_drc"] - frame[f"{endpoint}_pred"]
    return frame.reindex(names).to_numpy(np.float32), names.isin(frame.index).to_numpy()


def fold_scores(
    model: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    threads: int,
    seed: int,
    pca: int = PCA_COMPONENTS,
    n_extra: int = 0,
) -> np.ndarray:
    """Positive-class probabilities of ``test_x`` from a classifier fitted on the training fold. The
    last ``n_extra`` columns (stacked potencies) skip the PCA of the foundation models."""
    if model == "lightgbm":
        classifier = lgb.LGBMClassifier(
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
        classifier.fit(train_x, train_y)
        return classifier.predict_proba(test_x)[:, 1]
    # The foundation models read the set as the regression ones do: impute, standardise, PCA when
    # wider than PCA_COMPONENTS, fitted on the training fold.
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def prepared(width: int) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        if width > pca:
            steps.append(("pca", PCA(n_components=pca, random_state=seed)))
        return Pipeline(steps)

    split = train_x.shape[1] - n_extra
    reducer = prepared(split).fit(train_x[:, :split])
    blocks = [[reducer.transform(x[:, :split]).astype(np.float32)] for x in (train_x, test_x)]
    if n_extra:
        extra = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        ).fit(train_x[:, split:])
        for block, x in zip(blocks, (train_x, test_x), strict=True):
            block.append(extra.transform(x[:, split:]).astype(np.float32))
    fill = np.hstack(blocks[0]), np.hstack(blocks[1])
    if model == "tabpfn":
        import torch
        from tabpfn import TabPFNClassifier

        classifier = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        classifier.fit(fill[0], train_y)
        return classifier.predict_proba(fill[1])[:, 1]
    with tempfile.TemporaryDirectory() as directory:
        source, target = Path(directory) / "context.npz", Path(directory) / "prediction.npz"
        np.savez(source, x__tdi=fill[0], y__tdi=train_y.astype(float), query=fill[1])
        settings = {
            "task": "classification",
            "fine_tune": False,
            "fine_tune_steps": 0,
            "seed": seed,
        }
        settings["device"] = "cuda"
        command = [
            os.environ.get("MITRA_PYTHON", MITRA_PYTHON),
            str(MITRA_WORKER),
            "--source",
            str(source),
            "--target",
            str(target),
            "--settings",
            json.dumps(settings),
        ]
        subprocess.run(command, check=True)
        with np.load(target) as data:
            return data["prediction__tdi"]


def out_of_fold_scores(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    threads: int,
    seed: int,
    model: str = "lightgbm",
    pca: int = PCA_COMPONENTS,
    n_extra: int = 0,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities of a classifier; NaN where unlabelled."""
    scores = np.full(len(labels), np.nan)
    labelled = ~np.isnan(labels)
    for fold in np.unique(folds):
        train = labelled & (folds != fold)
        test = labelled & (folds == fold)
        scores[test] = fold_scores(
            model,
            features[train],
            labels[train].astype(int),
            features[test],
            threads,
            seed,
            pca,
            n_extra,
        )
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
    parser.add_argument("--model", choices=MODELS, default="lightgbm")
    parser.add_argument("--features", choices=tuple(FEATURE_SETS), default="fp_dense")
    parser.add_argument(
        "--pca",
        type=int,
        default=PCA_COMPONENTS,
        help="components the foundation models keep of a wider set (the regression models: 500)",
    )
    parser.add_argument(
        "--stack-bundle",
        type=Path,
        help="run bundle of a graph cell: add every head of its out-of-fold predictions "
        "(out_of_fold.parquet and out_of_fold_auxiliary.parquet) as extra columns",
    )
    parser.add_argument(
        "--embedding-bundle",
        type=Path,
        help="run bundle of a cell run with save_embedding: add its out-of-fold encodings",
    )
    parser.add_argument(
        "--extra-table",
        type=Path,
        help="parquet with Molecule_Name and extra feature columns (e.g. predictions of a model "
        "trained on another TDI source), added to every enzyme's features",
    )
    parser.add_argument(
        "--coverage-of",
        type=Path,
        help="run bundle whose compounds define the covered set without adding its columns",
    )
    parser.add_argument(
        "--covered-only",
        action="store_true",
        help="score only the compounds the bundle's folds covered (the direct-labelled ones)",
    )
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
    if args.model == "tabpfn":
        from cypshift.local_env import load_local_env

        load_local_env()  # the TabPFN token, never printed

    table = pd.read_csv(args.table)
    smiles = table["SMILES"].tolist()
    logger.info("featurising %d compounds", len(smiles))
    fingerprint, dense, embedding = FEATURE_SETS[args.features]
    features = molecular_feature_matrix(
        smiles,
        n_bits=args.bits if args.model == "lightgbm" else 2048,
        fingerprint_radius=2,
        physchem=dense,
        mechanism=dense,
        ionization_geometry=False,
        ionization_cache="data/external/ionization/features.parquet",
        molecular_embedding=embedding,
        use_fingerprint=fingerprint,
    )
    logger.info("%s features: %d columns", args.features, features.shape[1])
    covered = np.ones(len(table), dtype=bool)
    if args.coverage_of:
        seen = pd.read_parquet(args.coverage_of / "out_of_fold.parquet")["Molecule_Name"]
        covered &= table["Molecule_Name"].isin(seen).to_numpy()
    if args.embedding_bundle:
        encodings, seen = bundle_features(table, args.embedding_bundle, embedding=True)
        features = np.hstack([features, encodings])
        covered &= seen
    extra = []
    if args.predicted_potency:
        extra.append(predicted_potency(table, args.predicted_potency))
    if args.extra_table:
        frame = pd.read_parquet(args.extra_table).set_index("Molecule_Name")
        extra.append(frame.reindex(table["Molecule_Name"]).to_numpy(np.float32))
    if args.stack_bundle:
        stacked, seen = bundle_features(table, args.stack_bundle, embedding=False)
        extra.append(stacked)
        covered &= seen
    features = np.hstack([features, *extra])
    n_extra = sum(block.shape[1] for block in extra) + (1 if args.direct_potency else 0)
    labels = {e: table[f"{e}_is_TDI"].astype(float) for e in ENZYMES}
    if args.covered_only:
        labels = {e: label.where(covered) for e, label in labels.items()}
        logger.info("scoring the %d compounds the bundles cover", int(covered.sum()))
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
        scores = out_of_fold_scores(
            columns, truth, folds, args.threads, args.seed, args.model, args.pca, n_extra
        )
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
