"""Similarity computations and the matched-holdout selection."""

import numpy as np
import pytest

from cypshift.data.splits import (
    leave_one_out_max_similarity,
    max_similarity_against_retained,
    max_similarity_to,
    morgan_fingerprints,
    similarity_matched_holdout,
    similarity_profile_gap,
)

# A short congeneric series plus unrelated scaffolds, so similarities are not degenerate.
SMILES = [
    "CCOc1ccccc1",
    "CCOc1ccc(C)cc1",
    "CCOc1ccc(Cl)cc1",
    "CCOc1ccc(Br)cc1",
    "c1ccc2[nH]ccc2c1",
    "c1ccc2ncccc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "OCC1OC(O)C(O)C(O)C1O",
    "c1ccncc1",
    "CCCCCCCC",
    "NC(=O)c1ccccc1",
    "Cc1ccc(S(N)(=O)=O)cc1",
    "OC(=O)c1ccccc1",
]


# Stand-ins for the blinded test set: related to the training series but not present in it.
HELD_OUT_LIKE_TEST = [
    "CCOc1ccc(F)cc1",
    "CCOc1ccc(I)cc1",
    "CCOc1ccc(CC)cc1",
    "c1ccc2occc2c1",
]


def test_unparseable_smiles_raises_rather_than_silently_dropping():
    with pytest.raises(ValueError, match="could not parse SMILES at position 1"):
        morgan_fingerprints(["CCO", "not_a_molecule"])


def test_a_molecule_is_maximally_similar_to_itself():
    fingerprints = morgan_fingerprints(SMILES[:3])
    assert max_similarity_to(fingerprints, fingerprints) == pytest.approx(1.0)


def test_leave_one_out_excludes_the_compound_itself():
    fingerprints = morgan_fingerprints(SMILES)
    scores = leave_one_out_max_similarity(fingerprints)
    assert (scores < 1.0).all()
    assert (scores >= 0.0).all()


def test_similarity_against_retained_excludes_self_for_retained_compounds():
    fingerprints = morgan_fingerprints(SMILES)
    retained = np.array([True] * 7 + [False] * 7)
    scores = max_similarity_against_retained(fingerprints, retained)
    assert (scores[retained] < 1.0).all()


def test_holdout_has_the_requested_size_and_is_reproducible():
    train, test = SMILES, HELD_OUT_LIKE_TEST
    first = similarity_matched_holdout(train, test, holdout_fraction=0.5, n_bins=2, n_rounds=2)
    second = similarity_matched_holdout(train, test, holdout_fraction=0.5, n_bins=2, n_rounds=2)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == bool and len(first) == len(train)
    assert first.sum() == round(0.5 * len(train))


def test_a_different_seed_selects_different_compounds():
    train, test = SMILES, HELD_OUT_LIKE_TEST
    a = similarity_matched_holdout(train, test, holdout_fraction=0.5, n_bins=2, seed=0)
    b = similarity_matched_holdout(train, test, holdout_fraction=0.5, n_bins=2, seed=7)
    assert not np.array_equal(a, b)


def test_profile_gap_is_zero_for_identical_distributions():
    values = np.linspace(0.2, 0.9, 50)
    for gap in similarity_profile_gap(values, values).values():
        assert gap == pytest.approx(0.0)


def test_profile_gap_is_signed():
    low, high = np.linspace(0.1, 0.4, 50), np.linspace(0.5, 0.8, 50)
    assert similarity_profile_gap(low, high)["q50_gap"] < 0
    assert similarity_profile_gap(high, low)["q50_gap"] > 0


def test_bin_selection_hits_the_exact_count_even_when_rounding_overshoots():
    """Independent per-bin rounding can sum past (or short of) the target; the result must
    still hold out exactly n_holdout compounds."""
    from cypshift.data.splits import _select_by_bins

    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 200)
    edges = np.array([-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf])
    # Shares that each round up (0.17 * 97 ~= 16.5) so the naive sum overshoots.
    target_share = np.full(5, 0.2)
    for n_holdout in (97, 50, 3):
        mask = _select_by_bins(scores, edges, target_share, n_holdout, 0.5, rng)
        assert int(mask.sum()) == n_holdout
