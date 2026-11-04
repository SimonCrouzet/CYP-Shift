"""Write a TDI-track submission from graph cells and tabular classifiers, one blend per enzyme.

    python scripts/tdi_submission.py --rev questions3-m1 \\
        --names focus_tdi5_label_b_reg_uncovered_0_8352aa --output submissions/tdi.csv

    python scripts/tdi_submission.py --rev questions3-m1 --output submissions/tdi_blend.csv \\
        --members CYP2D6=graph:NAME_A+graph:NAME_B@0,1,2 \\
        --members CYP3A4=graph:NAME_A+table:data/tdi_runs/oof.csv,data/tdi_runs/test.csv

``--names`` gives both enzymes the label heads of those graph cells. ``--members ENZYME=SPEC+SPEC``
replaces an enzyme's members; a spec is one of:

* ``graph:NAME[@SEEDS]``: the ``is_TDI`` label head of a graph cell run on the TDI folds. The
  score of a compound is the mean of the five fold models on the test set, and its out-of-fold
  score is the one from the model that did not train on it. ``SEEDS`` (comma-separated, default 0)
  averages the cell's runs of those seeds.
* ``table:OOF,TEST``: the out-of-fold scores and the test scores written by ``tdi_baseline.py``
  (``--output`` and ``--test-output``), their ``<ENZYME>_score`` column.

The members live on different score scales, so each score is replaced by its place in that
member's own out-of-fold scores of the scored compounds (those with a direct measurement and a
label), and the members' places are averaged with equal weight. The call is then a threshold on
that blend. ``oof`` (default) takes the blend value at which the out-of-fold blend flags the
scored compounds' prevalence and applies it to the test set, so the number of test calls follows
the test scores; ``share`` flags the same share of the test set instead.

The report gives per enzyme the out-of-fold MCC and AUC of the blend, the same two within the
compounds the frozen TDI folds mark as in the test's similarity range (``--folds-file``; the MCC
uses the threshold of the whole scored set), and the number of test calls. A JSON next to the CSV
records the members and the numbers.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cypshift.data.challenge import load  # noqa: E402 -- after the src path insert

logger = logging.getLogger("tdi_submission")

BUNDLES = Path("data/run_bundles")
ENZYMES = ("CYP2D6", "CYP3A4")
FOLDS_FILE = Path("data/tdi_runs/folds_tdi5_seed0.parquet")


@dataclass(frozen=True)
class Member:
    """One scorer of an enzyme: the spec it was written as, and where its scores are."""

    spec: str
    kind: str
    name: str = ""
    seeds: tuple[int, ...] = (0,)
    oof_path: Path | None = None
    test_path: Path | None = None


def label_column(enzyme: str) -> str:
    return f"{enzyme}_pIC50_direct_inhibition__tdi_label"


def parse_member(spec: str) -> Member:
    """A :class:`Member` from ``graph:NAME[@SEEDS]`` or ``table:OOF,TEST``."""
    kind, _, rest = spec.partition(":")
    if kind == "graph" and rest:
        name, _, seeds = rest.partition("@")
        return Member(spec, kind, name, tuple(int(s) for s in seeds.split(",")) if seeds else (0,))
    if kind == "table" and rest.count(",") == 1:
        oof, test = rest.split(",")
        return Member(spec, kind, oof_path=Path(oof), test_path=Path(test))
    raise ValueError(f"member {spec!r} is graph:NAME[@SEEDS] or table:OOF,TEST")


def parse_members(items: list[str]) -> dict[str, list[Member]]:
    """``{enzyme: members}`` from ``ENZYME=SPEC+SPEC`` items."""
    parsed = {}
    for item in items:
        enzyme, _, specs = item.partition("=")
        if enzyme not in ENZYMES or not specs:
            raise ValueError(f"--members {item!r} is ENZYME=SPEC+SPEC with ENZYME in {ENZYMES}")
        parsed[enzyme] = [parse_member(spec) for spec in specs.split("+")]
    return parsed


def member_scores(member: Member, enzyme: str, rev: str) -> tuple[pd.Series, pd.Series]:
    """The out-of-fold scores and the mean fold-model test scores of a member, by compound."""
    if member.kind == "table":
        column = f"{enzyme}_score"
        oof = pd.read_csv(member.oof_path).set_index("Molecule_Name")[column]
        test = pd.read_csv(member.test_path).set_index("Molecule_Name")[column]
        return oof.dropna(), test
    oof_parts, test_parts = [], []
    for seed in member.seeds:
        path = BUNDLES / rev / f"reg_{member.name}_seed{seed}"
        oof_table = pd.read_parquet(path / "out_of_fold_auxiliary.parquet")
        test_table = pd.read_parquet(path / "test_by_fold_auxiliary.parquet")
        oof_parts.append(oof_table.set_index("Molecule_Name")[label_column(enzyme)])
        test_parts.append(test_table.groupby("Molecule_Name")[label_column(enzyme)].mean())
    # Another seed has other folds but still scores each compound out of fold once; average them.
    return (
        pd.concat(oof_parts, axis=1).mean(axis=1),
        pd.concat(test_parts, axis=1).mean(axis=1),
    )


def places(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The share of ``reference`` at or below each of ``values``."""
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def blend(oofs: list[np.ndarray], tests: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The mean place of each compound among its member's out-of-fold scores, out of fold and on
    the test set. ``oofs[i]`` and ``tests[i]`` are member ``i``'s scores."""
    oof = np.mean([places(o, o) for o in oofs], axis=0)
    test = np.mean([places(o, t) for o, t in zip(oofs, tests, strict=True)], axis=0)
    return oof, test


def calls(
    oof: np.ndarray, truth: np.ndarray, test: np.ndarray, rule: str = "oof"
) -> tuple[np.ndarray, float]:
    """Boolean test calls, and the score threshold they use.

    ``oof`` and ``truth`` are the out-of-fold scores and 0/1 labels of the scored compounds. The
    prevalence of ``truth`` is the share flagged: ``oof`` applies the score that flags that share
    of ``oof`` to ``test``; ``share`` flags that share of ``test``.
    """
    prevalence = float(np.mean(truth))
    if rule == "oof":
        threshold = float(np.quantile(oof, 1.0 - prevalence))
    elif rule == "share":
        threshold = float(np.quantile(test, 1.0 - prevalence))
    else:
        raise ValueError(f"rule is 'oof' or 'share', not {rule!r}")
    return test >= threshold, threshold


def out_of_fold_report(
    oof: np.ndarray, truth: np.ndarray, subset: np.ndarray | None = None
) -> dict[str, float]:
    """MCC and AUC of out-of-fold scores, the call flagging the prevalence share of all of them;
    with ``subset`` (a boolean mask) both are measured within those compounds only."""
    called, _ = calls(oof, truth, oof, "share")
    if subset is None:
        subset = np.ones(len(truth), dtype=bool)
    y = truth[subset].astype(int)
    report = {"n": int(subset.sum()), "positives": int(y.sum())}
    if len(set(y)) < 2:
        return report
    report["MCC"] = float(matthews_corrcoef(y, called[subset].astype(int)))
    report["AUC"] = float(roc_auc_score(y, oof[subset]))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the graph cells' results namespace")
    parser.add_argument("--names", default="", help="comma-separated graph cells for both enzymes")
    parser.add_argument("--members", action="append", default=[], help="ENZYME=SPEC+SPEC")
    parser.add_argument("--rule", choices=("oof", "share"), default="oof")
    parser.add_argument("--directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--folds-file", type=Path, default=FOLDS_FILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    members = parse_members(args.members)
    for enzyme in ENZYMES:
        if enzyme not in members:
            if not args.names:
                parser.error(f"no members for {enzyme}: give --names or --members {enzyme}=...")
            members[enzyme] = [parse_member(f"graph:{n}") for n in args.names.split(",")]

    test = load("test", args.directory)
    labels = load("tdi", args.directory).set_index("Molecule_Name")
    direct = set(load("inhibition", args.directory)["Molecule_Name"])
    in_range = None
    if args.folds_file.exists():
        in_range = pd.read_parquet(args.folds_file).set_index("Molecule_Name")["in_test_range"]
    submission = test[["SMILES", "Molecule_Name"]].copy()
    summary = {"rule": args.rule, "enzymes": {}}
    for enzyme in ENZYMES:
        scores = [member_scores(m, enzyme, args.rev) for m in members[enzyme]]
        common = set.intersection(*(set(oof.index) for oof, _ in scores))
        truth = labels[f"{enzyme}_is_TDI"]
        scored = [n for n in truth.index if n in common and n in direct and pd.notna(truth[n])]
        oof_block = [oof.loc[scored].to_numpy() for oof, _ in scores]
        test_block = [t.reindex(test["Molecule_Name"]).to_numpy() for _, t in scores]
        if any(np.isnan(t).any() for t in test_block):
            raise ValueError(f"a {enzyme} member has no test score for some compounds")
        oof_blend, test_blend = blend(oof_block, test_block)
        y = truth.loc[scored].to_numpy().astype(int)
        called, threshold = calls(oof_blend, y, test_blend, args.rule)
        submission[f"{enzyme}_is_TDI"] = called
        subset = (
            None
            if in_range is None
            else in_range.reindex(scored).fillna(False).to_numpy(dtype=bool)
        )
        summary["enzymes"][enzyme] = {
            "members": [m.spec for m in members[enzyme]],
            "scored_compounds": len(scored),
            "prevalence": float(y.mean()),
            "all": out_of_fold_report(oof_blend, y),
            "test_range": None if subset is None else out_of_fold_report(oof_blend, y, subset),
            "threshold": threshold,
            "test_calls": int(called.sum()),
            "test_share": float(called.mean()),
        }
        logger.info("%s: %s", enzyme, json.dumps(summary["enzymes"][enzyme]))
    assert submission[["CYP2D6_is_TDI", "CYP3A4_is_TDI"]].notna().all().all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %d rows to %s", len(submission), args.output)


if __name__ == "__main__":
    main()
