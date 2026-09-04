"""Pocket descriptors computed from a crystal structure."""

import numpy as np
import pytest

from cypshift.data.pocket import (
    ACIDIC,
    BASIC,
    HYDROPATHY,
    POCKET_DESCRIPTOR_NAMES,
    PocketComposition,
    bound_ligand_atoms,
    heme_iron,
)


def _composition(residues, distances=None):
    distances = distances or [3.5] * len(residues)
    return PocketComposition("TEST", tuple(residues), tuple(distances))


def test_the_descriptor_has_one_entry_per_declared_name():
    assert len(_composition(["ASP", "PHE"]).descriptor()) == len(POCKET_DESCRIPTOR_NAMES)


def test_an_acidic_pocket_reads_as_negatively_charged():
    """The property that distinguishes CYP2D6, which pairs a protonated amine."""
    descriptor = _composition(["ASP", "GLU", "ALA", "LEU"]).descriptor()
    assert descriptor[POCKET_DESCRIPTOR_NAMES.index("net_charge")] < 0
    assert descriptor[POCKET_DESCRIPTOR_NAMES.index("fraction_acidic")] == pytest.approx(0.5)


def test_a_basic_pocket_reads_as_positively_charged():
    """And this distinguishes CYP2C9, which pairs an anion."""
    descriptor = _composition(["ARG", "LYS", "ALA", "LEU"]).descriptor()
    assert descriptor[POCKET_DESCRIPTOR_NAMES.index("net_charge")] > 0


def test_opposite_pockets_get_opposite_charges():
    acidic = _composition(["ASP", "GLU", "ALA"]).descriptor()
    basic = _composition(["ARG", "LYS", "ALA"]).descriptor()
    index = POCKET_DESCRIPTOR_NAMES.index("net_charge")
    assert np.sign(acidic[index]) == -np.sign(basic[index])


def test_an_aromatic_pocket_is_recognised():
    descriptor = _composition(["PHE", "PHE", "TYR", "ALA"]).descriptor()
    assert descriptor[POCKET_DESCRIPTOR_NAMES.index("fraction_aromatic")] == pytest.approx(0.75)


def test_hydropathy_separates_greasy_from_polar_pockets():
    greasy = _composition(["ILE", "LEU", "VAL"]).descriptor()
    polar = _composition(["ASP", "ARG", "ASN"]).descriptor()
    index = POCKET_DESCRIPTOR_NAMES.index("mean_hydropathy")
    assert greasy[index] > polar[index]


def test_an_empty_pocket_does_not_divide_by_zero():
    descriptor = _composition([]).descriptor()
    assert np.isfinite(descriptor).all()


def test_the_residue_groups_are_disjoint_where_they_should_be():
    assert ACIDIC.isdisjoint(BASIC)
    assert set(HYDROPATHY) >= ACIDIC and set(HYDROPATHY) >= BASIC


def test_a_structure_without_a_heme_is_rejected(tmp_path):
    path = tmp_path / "nothing.pdb"
    path.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00\n")
    with pytest.raises(ValueError, match="no heme iron"):
        heme_iron(path)


def test_a_structure_without_a_ligand_is_rejected(tmp_path):
    """Solvent and the cofactor must not be mistaken for a bound ligand."""
    path = tmp_path / "apo.pdb"
    path.write_text(
        "HETATM    1  O   HOH A 500       1.000   1.000   1.000  1.00  0.00\n"
        "HETATM    2 FE   HEM A 600       0.000   0.000   0.000  1.00  0.00\n"
    )
    with pytest.raises(ValueError, match="no bound ligand"):
        bound_ligand_atoms(path)


def test_a_tiny_het_group_is_not_a_pocket_defining_ligand(tmp_path):
    path = tmp_path / "small.pdb"
    path.write_text(
        "".join(
            f"HETATM  {i:3d}  C{i}  LIG A 700       {i}.000   0.000   0.000  1.00  0.00\n"
            for i in range(3)
        )
    )
    with pytest.raises(ValueError, match="only 3 atoms"):
        bound_ligand_atoms(path)
