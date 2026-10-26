"""Auxiliary row selection: group caps, determinism and the three selection methods."""

import pandas as pd
import pytest

from cypshift.data.selection import GROUP_COLUMNS, reduce_groups, select_rows

CORES = ["c1ccccc1{}", "c1ccncc1{}", "C1CCCCC1{}", "c1ccc2ccccc2c1{}", "C1CCOC1{}"]
TAILS = ["", "C", "CC", "CCC", "O", "CO", "N", "CN", "Cl", "F", "C(=O)O", "C(F)(F)F"]
SMILES = [core.format(tail) for core in CORES for tail in TAILS]


def _table(n_each: int = 30) -> pd.DataFrame:
    rows = []
    for group, (target, source) in enumerate([("CYP2D6", "novartis"), ("CYP3A4", "tox21")]):
        for smiles in SMILES[:n_each]:
            rows.append(
                {
                    "SMILES": smiles,
                    "target": target,
                    "assay": "reversible_pic50",
                    "source": source,
                    "data_space": "near_aux",
                    "value": float(group),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["clusters", "maxmin", "nearest"])
def test_a_method_keeps_the_requested_number_of_distinct_rows(method):
    kept = select_rows(SMILES, 12, method, reference_smiles=SMILES[:6], seed=0)
    assert len(kept) == len(set(kept.tolist())) == 12
    assert kept.min() >= 0 and kept.max() < len(SMILES)


@pytest.mark.parametrize("method", ["clusters", "maxmin", "nearest"])
def test_selection_is_repeatable(method):
    first = select_rows(SMILES, 12, method, reference_smiles=SMILES[:6], seed=3)
    second = select_rows(SMILES, 12, method, reference_smiles=SMILES[:6], seed=3)
    assert first.tolist() == second.tolist()


def test_nearest_keeps_the_rows_most_similar_to_the_reference():
    kept = select_rows(SMILES, 3, "nearest", reference_smiles=["c1ccccc1CC"], seed=0)
    assert "c1ccccc1CC" in {SMILES[i] for i in kept}


def test_nearest_needs_a_reference():
    with pytest.raises(ValueError, match="reference"):
        select_rows(SMILES, 3, "nearest")


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        select_rows(SMILES, 3, "telepathy")


def test_only_groups_over_the_cap_are_reduced():
    table = _table(30)
    table = pd.concat(
        [table, table.iloc[:5].assign(target="CYP2C9", source="octant")], ignore_index=True
    )
    reduced = reduce_groups(table, 10, "maxmin", seed=0)
    sizes = reduced.groupby(GROUP_COLUMNS).size()
    assert sorted(sizes.tolist()) == [5, 10, 10]
    assert set(reduced.index) <= set(table.index)


def test_a_cap_above_every_group_changes_nothing():
    table = _table(30)
    assert reduce_groups(table, 100, "clusters", seed=0).equals(table)
