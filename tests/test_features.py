"""Featurisation blocks, including the mechanistic substructure flags."""

import numpy as np
import pytest

from cypshift.features.molecular import featurise, mechanistic, morgan_counts, physicochemical


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
