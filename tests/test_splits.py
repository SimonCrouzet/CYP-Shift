"""Similarity computations and the matched-holdout selection."""

import numpy as np
import pytest

from cypshift.data.splits import (
    AnalogueSplit,
    analogue_holdout,
    analogue_split,
    leave_one_out_max_similarity,
    max_similarity_against_retained,
    max_similarity_to,
    morgan_fingerprints,
    murcko_scaffold,
    similarity_matched_holdout,
    similarity_profile_gap,
    stratified_scaffold_folds,
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


def test_scaffold_folds_are_a_partition():
    """Every compound lands in exactly one fold, and all folds are used."""
    folds = stratified_scaffold_folds(SMILES, [0] * len(SMILES), n_folds=3, seed=0)
    assert len(folds) == len(SMILES)
    assert set(folds.tolist()) == {0, 1, 2}


def test_a_scaffold_never_spans_two_folds():
    """Compounds sharing a Bemis-Murcko scaffold must stay together, so an analog series
    cannot straddle the train/validation boundary."""
    folds = stratified_scaffold_folds(SMILES, [0] * len(SMILES), n_folds=3, seed=0)
    scaffolds = np.array([murcko_scaffold(s) for s in SMILES])
    for scaffold in np.unique(scaffolds):
        assert len(set(folds[scaffolds == scaffold])) == 1


def test_scaffold_folds_are_reproducible():
    one = stratified_scaffold_folds(SMILES, [0] * len(SMILES), n_folds=3, seed=0)
    two = stratified_scaffold_folds(SMILES, [0] * len(SMILES), n_folds=3, seed=0)
    assert np.array_equal(one, two)


def test_acyclic_compound_is_its_own_scaffold_group():
    """An acyclic compound has an empty Murcko scaffold; it falls back to its own SMILES so
    acyclic compounds do not all collapse into one group."""
    assert murcko_scaffold("CCCCCCCC") == "CCCCCCCC"
    assert murcko_scaffold("CCOc1ccccc1") == "c1ccccc1"


def test_analogue_holdout_holds_out_analogues_of_retained_anchors():
    """Every held-out compound has a retained neighbour above the threshold (it is an analogue),
    and the holdout is roughly the requested size."""
    mask = analogue_holdout(SMILES, holdout_fraction=0.3, min_similarity=0.4, seed=0)
    assert mask.dtype == bool and len(mask) == len(SMILES)
    held = int(mask.sum())
    assert 0 < held <= len(SMILES)
    # A held-out compound must have a retained neighbour at least as similar as the threshold;
    # otherwise it was placed without being anyone's analogue.
    fingerprints = morgan_fingerprints(SMILES)
    for index in np.flatnonzero(mask):
        to_retained = max_similarity_against_retained(fingerprints, ~mask)[index]
        assert to_retained >= 0.4


def test_analogue_holdout_leaves_a_compound_with_no_analogue_retained():
    """A lone, dissimilar compound has no neighbour above the threshold, so it is never held out."""
    smiles = ["CCOc1ccccc1", "CCOc1ccc(C)cc1", "CCOc1ccc(Cl)cc1", "[Na+].[Cl-]"]
    mask = analogue_holdout(smiles, holdout_fraction=0.25, min_similarity=0.5, seed=0)
    assert not mask[-1]  # the salt is unlike any aryl ether, so it stays retained


def test_analogue_holdout_is_reproducible():
    one = analogue_holdout(SMILES, holdout_fraction=0.3, seed=0)
    two = analogue_holdout(SMILES, holdout_fraction=0.3, seed=0)
    assert np.array_equal(one, two)


def test_analogue_split_records_each_held_out_compound_s_anchor():
    """F11: the split exposes the retained anchor per held-out compound, and the mask matches
    analogue_holdout. Each held-out anchor is retained and similar above the threshold."""
    split = analogue_split(SMILES, holdout_fraction=0.3, min_similarity=0.4, seed=0)
    mask = analogue_holdout(SMILES, holdout_fraction=0.3, seed=0)
    assert np.array_equal(split.holdout_mask, mask)
    fingerprints = morgan_fingerprints(SMILES)
    for index in np.flatnonzero(split.holdout_mask):
        anchor = split.anchor_of[index]
        assert anchor >= 0
        assert not split.holdout_mask[anchor]  # the anchor is retained
        assert max_similarity_to([fingerprints[index]], [fingerprints[anchor]])[0] >= 0.4
    assert (split.anchor_of[~split.holdout_mask] == -1).all()  # retained carry no anchor


def test_anchor_label_coverage_reports_same_endpoint_references():
    """F11: coverage is the fraction of held-out-and-labelled compounds whose anchor is also
    labelled on that endpoint."""
    split = AnalogueSplit(
        holdout_mask=np.array([False, False, True, True]),
        anchor_of=np.array([-1, -1, 0, 1]),
    )
    labels = np.zeros((4, 4), dtype=bool)
    labels[2, 0] = labels[0, 0] = True  # held-out 2 and its anchor 0 both labelled on endpoint 0
    labels[3, 0] = True  # held-out 3 labelled, but its anchor 1 is not
    coverage = split.anchor_label_coverage(labels)
    assert coverage[0] == pytest.approx(0.5)
    assert coverage[1] != coverage[1]  # NaN: no held-out label on endpoint 1


def test_require_labelled_anchor_drops_a_pair_with_no_shared_endpoint():
    """F11: with require_labelled_anchor, a neighbour whose anchor shares no measured endpoint is
    not held out (it would have no same-endpoint reference)."""
    smiles = ["CCO", "CCO"]  # identical, so each is the other's only neighbour
    labels = np.array([[True, False, False, False], [False, True, False, False]])
    without = analogue_split(smiles, holdout_fraction=0.5, min_similarity=0.5, seed=0)
    with_rule = analogue_split(
        smiles,
        holdout_fraction=0.5,
        min_similarity=0.5,
        labels=labels,
        require_labelled_anchor=True,
        seed=0,
    )
    assert without.holdout_mask.sum() == 1
    assert with_rule.holdout_mask.sum() == 0
