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


def test_the_recognition_chemistry_distinguishes_the_scored_enzymes():
    """The point of describing rather than indexing: opposite charges are visible."""
    assert ENZYMES["CYP2D6"].has_acidic_anchor  # pairs a protonated amine
    assert ENZYMES["CYP2C9"].has_basic_anchor  # pairs an anion
    assert not ENZYMES["CYP2D6"].has_basic_anchor
    assert not ENZYMES["CYP2C9"].has_acidic_anchor


def test_the_planar_slot_belongs_to_family_one():
    assert ENZYMES["CYP1A2"].is_narrow_planar
    assert not ENZYMES["CYP3A4"].is_narrow_planar


def test_pocket_volumes_are_ordered_as_the_literature_reports():
    volumes = [ENZYMES[n].pocket_volume_angstrom3 for n in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")]
    assert volumes == sorted(volumes)


def test_near_identical_enzymes_get_near_identical_descriptors():
    """CYP2C9 and CYP2C19 are about 91% identical, so their data should pool."""
    np.testing.assert_allclose(descriptor("CYP2C9"), descriptor("CYP2C19"))


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
    register(Enzyme("CYP_TEST", 500, True, False, False, True, 2, "fixture"))
    try:
        assert len(descriptor("CYP_TEST")) == descriptor_width()
        assert describe_all(["CYP2D6", "CYP_TEST"]).shape == (2, descriptor_width())
    finally:
        ENZYMES.pop("CYP_TEST")


def test_describing_a_list_gives_one_row_each():
    assert describe_all(list(ISOFORMS)).shape == (len(ISOFORMS), descriptor_width())
