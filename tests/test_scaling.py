"""Per-kind feature scaling."""

import numpy as np
import pytest

from cypshift.features.scaling import BlockScaler

RNG = np.random.default_rng(0)
N_COUNTS, N_DENSE, N_BINARY = 6, 3, 2


def _matrix(n=200):
    counts = RNG.poisson(0.1, (n, N_COUNTS)).astype(np.float32)  # sparse, mostly zero
    dense = RNG.normal(300, 80, (n, N_DENSE)).astype(np.float32)
    binary = RNG.integers(0, 2, (n, N_BINARY)).astype(np.float32)
    return np.hstack([counts, dense, binary])


def _scaler():
    return BlockScaler(n_count_columns=N_COUNTS, n_dense_columns=N_DENSE)


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        _scaler().transform(_matrix())


def test_sparsity_of_the_count_block_survives():
    """The whole point: a zero bit must stay zero, not become a non-zero value."""
    features = _matrix()
    scaled = _scaler().fit_transform(features)
    was_zero = features[:, :N_COUNTS] == 0
    assert (scaled[:, :N_COUNTS][was_zero] == 0).all()


def test_uniform_standardisation_would_destroy_that_sparsity():
    """Contrast case, documenting why this module exists."""
    features = _matrix()
    naive = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    was_zero = features[:, :N_COUNTS] == 0
    assert not (naive[:, :N_COUNTS][was_zero] == 0).all()


def test_counts_are_compressed_but_still_ordered():
    features = _matrix()
    scaled = _scaler().fit_transform(features)
    column = features[:, 0]
    if column.max() > 1:
        order_before = np.argsort(column)
        assert np.all(np.diff(scaled[order_before, 0]) >= -1e-6)


def test_dense_columns_are_centred_and_scaled():
    scaled = _scaler().fit_transform(_matrix(500))
    dense = scaled[:, N_COUNTS : N_COUNTS + N_DENSE]
    assert np.allclose(dense.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(dense.std(axis=0), 1.0, atol=1e-2)


def test_binary_columns_pass_through_untouched():
    features = _matrix()
    scaled = _scaler().fit_transform(features)
    np.testing.assert_array_equal(
        scaled[:, N_COUNTS + N_DENSE :], features[:, N_COUNTS + N_DENSE :]
    )


def test_the_fitted_scaler_applies_to_new_data():
    scaler = _scaler().fit(_matrix(300))
    other = _matrix(50)
    assert scaler.transform(other).shape == other.shape


def test_scaling_is_deterministic():
    features = _matrix()
    scaler = _scaler().fit(features)
    np.testing.assert_array_equal(scaler.transform(features), scaler.transform(features))
