"""Tests for the Z-scale encoding of pocket residues."""

import numpy as np
import pytest

from cypshift.data.zscales import (
    ZSCALE_DESCRIPTOR_NAMES,
    ZSCALES,
    pooled_zscales,
    residue_zscales,
)


def test_table_covers_the_twenty_standard_residues():
    assert len(ZSCALES) == 20
    assert all(len(v) == 3 for v in ZSCALES.values())


def test_hydrophobic_and_hydrophilic_residues_have_opposite_z1():
    """z1 rises with hydrophilicity, which is the scale's defining axis."""
    assert ZSCALES["ASP"][0] > 0 > ZSCALES["LEU"][0]
    assert ZSCALES["ARG"][0] > ZSCALES["ILE"][0]


def test_bulky_residues_have_larger_z2_than_small_ones():
    assert ZSCALES["TRP"][1] > ZSCALES["ALA"][1] > ZSCALES["GLY"][1]


def test_pooled_descriptor_has_the_declared_width():
    assert pooled_zscales(("ASP", "PHE", "LEU")).shape == (len(ZSCALE_DESCRIPTOR_NAMES),)


def test_empty_pocket_gives_zeros_rather_than_failing():
    assert not pooled_zscales(()).any()


def test_non_standard_residues_are_skipped_not_zero_filled():
    """A zero vector sits near glutamine, so filling with it would invent chemistry."""
    assert np.allclose(residue_zscales(("PHE", "HEM")), residue_zscales(("PHE",)))


def test_a_uniform_pocket_has_zero_spread():
    pooled = pooled_zscales(("LEU", "LEU", "LEU"))
    assert np.allclose(pooled[3:], 0.0)


def test_a_mixed_pocket_has_non_zero_spread():
    """Mean alone cannot tell a uniform pocket from a polar anchor in a greasy wall."""
    assert pooled_zscales(("ASP", "LEU")).__getitem__(slice(3, None)).min() > 0


def test_mean_is_the_average_of_the_member_residues():
    expected = np.mean([ZSCALES["ASP"], ZSCALES["GLU"]], axis=0)
    assert np.allclose(pooled_zscales(("ASP", "GLU"))[:3], expected, atol=1e-5)


def test_order_does_not_change_the_descriptor():
    """Pooling is over a set of contacts; the order they appear in the file is not
    chemistry."""
    forward = pooled_zscales(("ASP", "PHE", "LEU"))
    reverse = pooled_zscales(("LEU", "PHE", "ASP"))
    assert np.allclose(forward, reverse)


@pytest.mark.parametrize("residues", [("ASP", "GLU"), ("PHE", "TRP"), ("LEU",)])
def test_descriptor_is_finite(residues):
    assert np.isfinite(pooled_zscales(residues)).all()


def test_acidic_and_aromatic_pockets_are_distinguishable():
    """The point of the encoding: an anion-pairing pocket must not look like a
    stacking one."""
    acidic = pooled_zscales(("ASP", "GLU", "ASP"))
    aromatic = pooled_zscales(("PHE", "TRP", "PHE"))
    assert np.linalg.norm(acidic - aromatic) > 1.0
