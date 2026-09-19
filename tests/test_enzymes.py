"""Describing an enzyme rather than indexing it."""

import numpy as np
import pytest

from cypshift.config import ISOFORMS
from cypshift.data.enzymes import (
    DESCRIPTOR_FAMILIES,
    ENZYMES,
    Enzyme,
    describe_all,
    descriptor,
    descriptor_width,
    register,
)


def test_every_scored_enzyme_is_registered():
    assert set(ISOFORMS) <= set(ENZYMES)


def test_the_computed_composition_reflects_the_pocket_chemistry():
    """Composition is read off the real co-crystal contacts, not hand-set: CYP2D6's
    acidic anchor for a protonated amine is present, and its pocket has no basic residue."""
    assert ENZYMES["CYP2D6"].has_acidic  # the Asp/Glu that pairs a protonated amine
    assert not ENZYMES["CYP2D6"].has_basic
    assert ENZYMES["CYP2C9"].has_basic  # the Arg that pairs an anion


def test_family_number_is_definitional():
    assert ENZYMES["CYP1A2"].family == 1
    assert ENZYMES["CYP3A4"].family == 3


def test_a_promiscuous_pocket_computes_larger_than_a_tight_one():
    """CYP3A4's large cavity vs CYP1A2's tight slot -- the clearest ordering the computed
    volumes recover (the full literature ordering is not reproducible from single crystals)."""
    assert ENZYMES["CYP3A4"].fpocket_volume > ENZYMES["CYP1A2"].fpocket_volume
    assert all(ENZYMES[n].pykvfinder_volume >= 0 for n in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"))


def test_near_identical_enzymes_get_near_identical_sequence_descriptors():
    """CYP2C9 and CYP2C19 are about 91% identical, so their sequence-derived descriptors
    (which need only the alignment) nearly coincide and their data pools. The
    structure-derived sources differ because the two crystals differ."""
    from cypshift.data.conditioning import enzyme_descriptor

    np.testing.assert_allclose(
        enzyme_descriptor("CYP2C9", ("ctd",)), enzyme_descriptor("CYP2C19", ("ctd",)), atol=0.05
    )


def test_enzymes_with_opposite_chemistry_get_different_descriptors():
    assert not np.allclose(descriptor("CYP2C9"), descriptor("CYP2D6"))


def test_descriptor_families_can_be_selected():
    for family in DESCRIPTOR_FAMILIES:
        assert len(descriptor("CYP3A4", (family,))) == descriptor_width((family,))
    assert descriptor_width(DESCRIPTOR_FAMILIES) == sum(
        descriptor_width((f,)) for f in DESCRIPTOR_FAMILIES
    )


def test_an_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown descriptor family"):
        descriptor("CYP3A4", ("astrology",))


def test_an_unregistered_enzyme_is_rejected():
    with pytest.raises(KeyError, match="not registered"):
        descriptor("CYP_NOT_REAL")


def test_a_new_enzyme_needs_no_model_change():
    """The whole point: extension is a row, not an architecture edit."""
    register(Enzyme("CYP_TEST", 2, 500.0, 100.0, True, False, True, "fixture"))
    try:
        assert len(descriptor("CYP_TEST")) == descriptor_width()
        assert describe_all(["CYP2D6", "CYP_TEST"]).shape == (2, descriptor_width())
    finally:
        ENZYMES.pop("CYP_TEST")


def test_describing_a_list_gives_one_row_each():
    assert describe_all(list(ISOFORMS)).shape == (len(ISOFORMS), descriptor_width())
