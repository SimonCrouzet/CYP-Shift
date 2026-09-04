"""Tests for assembling an enzyme's conditioning vector from descriptor sources."""

import numpy as np
import pytest

from cypshift.data import enzymes
from cypshift.data.conditioning import (
    DESCRIPTOR_SOURCES,
    STRUCTURAL_SOURCES,
    covered_enzymes,
    descriptor_names,
    enzyme_descriptor,
    source_names,
)
from cypshift.data.pocket import STRUCTURES


def test_every_source_names_its_columns():
    for source in DESCRIPTOR_SOURCES:
        assert source_names(source)


def test_column_names_are_unique_within_a_source():
    """Two columns sharing a name would be indistinguishable in a sweep report."""
    for source in DESCRIPTOR_SOURCES:
        names = source_names(source)
        assert len(set(names)) == len(names)


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown descriptor source"):
        source_names("fingerprint")


def test_unknown_source_is_refused_when_building():
    with pytest.raises(ValueError, match="unknown descriptor sources"):
        enzyme_descriptor("CYP2D6", ("fingerprint",))


def test_names_concatenate_in_selection_order():
    assert descriptor_names(("curated", "ctd")) == source_names("curated") + source_names("ctd")


def test_curated_alone_needs_no_structure_or_sequence():
    """The default must work for an enzyme with neither, or registering one is a
    two-step affair."""
    assert len(enzyme_descriptor("CYP2B6", ("curated",))) == len(source_names("curated"))


def test_descriptor_width_matches_the_declared_names():
    descriptor = enzyme_descriptor("CYP2D6", ("curated", "srs"))
    assert len(descriptor) == len(descriptor_names(("curated", "srs")))


def test_a_structural_source_without_a_structure_fails_loudly():
    """Zero-filling would place the enzyme at a real point rather than a missing one."""
    missing = next(name for name in enzymes.ENZYMES if name not in STRUCTURES)
    with pytest.raises(ValueError, match="no registered structure"):
        enzyme_descriptor(missing, ("srs",))


def test_structural_sources_cover_only_the_enzymes_with_structures():
    for source in STRUCTURAL_SOURCES:
        assert covered_enzymes((source,)) == set(STRUCTURES)


def test_sequence_sources_reach_further_than_structural_ones():
    """Pretraining on auxiliaries is the point, and they have no co-crystal here."""
    assert len(covered_enzymes(("ctd",))) > len(covered_enzymes(("srs",)))


def test_coverage_of_a_selection_is_the_intersection():
    assert covered_enzymes(("ctd", "srs")) == covered_enzymes(("srs",))


def test_descriptor_is_finite():
    assert np.isfinite(enzyme_descriptor("CYP2D6", ("curated",))).all()


def test_two_enzymes_get_different_vectors():
    """A source that returned the same vector for every enzyme would condition on
    nothing."""
    first = enzyme_descriptor("CYP2C9", ("curated",))
    second = enzyme_descriptor("CYP2D6", ("curated",))
    assert not np.allclose(first, second)
