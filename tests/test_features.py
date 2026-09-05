"""Featurisation blocks, including the mechanistic substructure flags."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from cypshift.features.molecular import (
    basic_amine_geometry,
    featurise,
    mechanistic,
    morgan_counts,
    physicochemical,
)


def test_unparseable_smiles_raises_with_its_position():
    with pytest.raises(ValueError, match="position 1"):
        physicochemical(["CCO", "!!!"])


def test_basic_amine_is_detected_on_an_aliphatic_amine_only():
    frame = mechanistic(["CCN(CC)CC", "c1ccncc1", "CC(=O)Nc1ccccc1"])
    assert frame["has_basic_amine"].tolist() == [1.0, 0.0, 0.0]


def test_type_ii_flag_fires_on_heme_coordinating_aromatic_nitrogen():
    # Pyridine, imidazole and 1,2,4-triazole coordinate the iron; benzene cannot.
    frame = mechanistic(["c1ccncc1", "c1c[nH]cn1", "c1nc[nH]n1", "c1ccccc1"])
    assert frame["is_type_ii_candidate"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_an_imidazole_nh_nitrogen_does_not_count_as_coordinating():
    # Only one of imidazole's two nitrogens has a free in-plane lone pair; the NH
    # nitrogen's is delocalised into the ring and cannot bind the metal.
    assert mechanistic(["c1c[nH]cn1"])["n_heme_coordinating_n"].iloc[0] == 1.0
    # N-methylimidazole is the same case with the hydrogen replaced.
    assert mechanistic(["Cn1ccnc1"])["n_heme_coordinating_n"].iloc[0] == 1.0


def test_carboxylic_acid_is_counted():
    frame = mechanistic(["OC(=O)c1ccccc1", "CCO"])
    assert frame["n_carboxylic_acids"].tolist() == [1.0, 0.0]


def test_physicochemical_returns_one_column_per_property():
    frame = physicochemical(["CCO", "c1ccccc1"])
    assert len(frame) == 2
    assert "clogp" in frame.columns and "molecular_weight" in frame.columns
    assert frame["molecular_weight"].iloc[1] > frame["molecular_weight"].iloc[0]


def test_fingerprint_width_is_the_requested_size():
    assert morgan_counts(["CCO", "c1ccccc1"], n_bits=512).shape == (2, 512)


def test_featurise_concatenates_the_requested_blocks():
    smiles = ["CCO", "c1ccncc1"]
    fingerprint_only = featurise(smiles, physchem=False, mechanism=False, n_bits=256)
    everything = featurise(smiles, n_bits=256)
    assert fingerprint_only.shape == (2, 256)
    assert everything.shape[1] == 256 + len(physicochemical(smiles).columns) + len(
        mechanistic(smiles).columns
    )


def test_requesting_no_blocks_is_an_error():
    with pytest.raises(ValueError, match="at least one feature block"):
        featurise(["CCO"], fingerprint=False, physchem=False, mechanism=False)


def test_features_are_finite():
    assert np.isfinite(featurise(["CCO", "c1ccncc1", "CCN(CC)CC"], n_bits=256)).all()


class TestBasicAmineGeometry:
    """CYP2D6 recognises a basic nitrogen ion-paired to Asp301 or Glu216 with an
    aromatic ring stacked nearby. Counting amines cannot see the arrangement."""

    @staticmethod
    def _frame(*smiles: str) -> pd.DataFrame:
        return basic_amine_geometry([Chem.MolFromSmiles(s) for s in smiles])

    def test_a_molecule_without_a_basic_amine_is_marked_absent(self):
        """-1 rather than 0: zero bonds would mean the nitrogen is on the ring."""
        row = self._frame("CC(=O)Oc1ccccc1C(=O)O").iloc[0]
        assert row["min_basic_n_to_aromatic_bonds"] == -1.0
        assert row["basic_n_in_pharmacophore_window"] == 0.0

    def test_a_molecule_without_an_aromatic_ring_is_marked_absent(self):
        assert self._frame("CCN(CC)CC").iloc[0]["min_basic_n_to_aromatic_bonds"] == -1.0

    def test_a_known_substrate_falls_inside_the_window(self):
        """Diphenhydramine and codeine are both CYP2D6 substrates."""
        frame = self._frame("CN(C)CCOC(c1ccccc1)c1ccccc1", "CN1CCC23C4Oc5c(O)ccc(CC1C2C=CC4O)c35")
        assert (frame["basic_n_in_pharmacophore_window"] == 1.0).all()

    def test_distance_is_the_shortest_path_to_any_aromatic_atom(self):
        """Benzylamine: N-C-c, so two bonds."""
        assert self._frame("NCc1ccccc1").iloc[0]["min_basic_n_to_aromatic_bonds"] == 2.0

    def test_a_longer_linker_gives_a_longer_distance(self):
        short = self._frame("NCc1ccccc1").iloc[0]["min_basic_n_to_aromatic_bonds"]
        long = self._frame("NCCCCc1ccccc1").iloc[0]["min_basic_n_to_aromatic_bonds"]
        assert long > short

    def test_a_distance_outside_the_window_is_not_flagged(self):
        assert self._frame("NCc1ccccc1").iloc[0]["basic_n_in_pharmacophore_window"] == 0.0

    def test_rings_are_counted_not_atoms(self):
        """Diphenhydramine has two aromatic rings reachable from its amine."""
        assert (
            self._frame("CN(C)CCOC(c1ccccc1)c1ccccc1").iloc[0]["n_aromatic_rings_near_basic_n"]
            == 2.0
        )

    def test_the_geometry_columns_reach_the_mechanistic_block(self):
        frame = mechanistic(["CN(C)CCOC(c1ccccc1)c1ccccc1"])
        assert "basic_n_in_pharmacophore_window" in frame.columns

    def test_every_value_is_finite(self):
        frame = self._frame("CN(C)CCOC(c1ccccc1)c1ccccc1", "CC(=O)O", "c1ccccc1")
        assert np.isfinite(frame.to_numpy()).all()
