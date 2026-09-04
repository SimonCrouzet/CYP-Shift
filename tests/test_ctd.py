"""Tests for the composition and transition descriptors."""

import numpy as np
import pytest

from cypshift.data.ctd import (
    CTD_DESCRIPTOR_NAMES,
    PROPERTIES,
    composition,
    ctd_descriptor,
    transition,
)


def test_the_descriptor_has_the_published_width():
    """Lapins 2013: 21 composition plus 21 transition."""
    assert len(CTD_DESCRIPTOR_NAMES) == 42
    assert len(ctd_descriptor("MKVLLAVFGRT")) == 42


@pytest.mark.parametrize("name,classes", PROPERTIES.items())
def test_every_property_partitions_the_twenty_amino_acids(name, classes):
    """Classes must not overlap, or a residue would be counted twice."""
    joined = "".join(classes)
    assert len(joined) == len(set(joined)) == 20


def test_composition_sums_to_one():
    assert composition("MKVLLAVFGRT", PROPERTIES["charge"]).sum() == pytest.approx(1.0)


def test_composition_counts_the_right_class():
    """Charge: K and R positive, D and E negative, the rest neutral."""
    assert np.allclose(composition("KRKR", PROPERTIES["charge"]), [1.0, 0.0, 0.0])
    assert np.allclose(composition("DEDE", PROPERTIES["charge"]), [0.0, 0.0, 1.0])


def test_a_uniform_sequence_has_no_transitions():
    assert not transition("KRKRKR", PROPERTIES["charge"]).any()


def test_alternating_classes_transition_every_step():
    """K is positive, D negative, so KDKD crosses the 1-3 pair at every step."""
    values = transition("KDKD", PROPERTIES["charge"])
    assert values[1] == pytest.approx(1.0)  # pair (0, 2)
    assert values[0] == values[2] == 0.0


def test_transition_is_undirected():
    """Dubchak counts a step between two classes once, whichever way it runs."""
    assert np.allclose(
        transition("KD", PROPERTIES["charge"]), transition("DK", PROPERTIES["charge"])
    )


def test_unclassified_residues_are_skipped():
    """X is not one of the twenty and must not shift the composition."""
    assert np.allclose(
        composition("KRXKR", PROPERTIES["charge"]), composition("KRKR", PROPERTIES["charge"])
    )


def test_an_empty_sequence_is_refused():
    with pytest.raises(ValueError, match="empty sequence"):
        ctd_descriptor("")


def test_a_single_residue_has_no_transitions():
    assert not transition("K", PROPERTIES["charge"]).any()


def test_composition_block_sums_to_one_per_property():
    descriptor = ctd_descriptor("MKVLLAVFGRTDEQNWY")
    assert np.allclose(descriptor[:21].reshape(7, 3).sum(axis=1), 1.0)


def test_descriptor_is_finite():
    assert np.isfinite(ctd_descriptor("MKVLLAVFGRT")).all()
