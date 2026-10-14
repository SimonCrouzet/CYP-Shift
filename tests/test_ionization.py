"""Ionization state and conformer pharmacophore geometry at pH 7.4 (local; no network)."""

import numpy as np

from cypshift.features.ionization import (
    IONIZATION_COLUMNS,
    _compute_one,
    ionization_block,
    ionization_features,
)
from cypshift.features.molecular import featurise

AMINE = "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"  # a protonatable secondary amine with aromatics
ACID = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin: a carboxylate at pH 7.4, no cation
NEUTRAL = "CCO"


def test_basic_amine_is_cationic_with_a_finite_cation_aromatic_distance():
    n_cationic, n_anionic, distance = _compute_one(AMINE)
    assert n_cationic >= 1.0
    assert n_anionic == 0.0
    assert distance > 0.0  # a real through-space distance, not the -1 sentinel


def test_acid_is_anionic_and_has_no_cation_distance():
    n_cationic, n_anionic, distance = _compute_one(ACID)
    assert n_cationic == 0.0
    assert n_anionic >= 1.0
    assert distance == -1.0


def test_neutral_molecule_has_no_charge_and_no_distance():
    assert _compute_one(NEUTRAL) == (0.0, 0.0, -1.0)


def test_features_are_cached_and_reused(tmp_path):
    cache = tmp_path / "ion.parquet"
    first = ionization_features([AMINE, ACID, NEUTRAL], cache)
    assert cache.exists()
    assert first.shape == (3, len(IONIZATION_COLUMNS))
    # A second call reads the cache rather than recomputing, and returns the same values.
    second = ionization_features([AMINE, ACID, NEUTRAL], cache)
    assert np.allclose(first, second)


def test_featurise_appends_the_block(tmp_path):
    cache = tmp_path / "ion.parquet"
    base = featurise([AMINE, ACID], n_bits=256)
    appended = featurise([AMINE, ACID], n_bits=256, ionization=ionization_block(cache))
    assert appended.shape[1] - base.shape[1] == len(IONIZATION_COLUMNS)
