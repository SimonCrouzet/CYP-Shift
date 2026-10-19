"""The tabular foundation model's feature path: the shared set, compressed to fit TabPFN.

These tests cover the modular contract -- that the TFM assembles the same configured feature
set as the other models and only adapts its dimensionality -- without needing TabPFN itself
(the per-endpoint fit, which downloads weights and wants a token, is not exercised here).
"""

from cypshift.models.tfm import TabularFoundationModel

SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN(CC)CC",
    "c1ccncc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "CCCCCCCC",
    "c1ccc2ccccc2c1",
    "OCC1OC(O)C(O)C(O)C1O",
    "CC(N)C(=O)O",
    "c1ccc(cc1)O",
    "ClCCl",
    "FC(F)F",
    "CCOCC",
    "CC#N",
    "C=CC=C",
    "CC(C)=O",
]


def test_raw_features_include_the_fingerprint():
    """The TFM assembles the fingerprint like every other model -- not a dense-only set."""
    model = TabularFoundationModel(n_bits=256, physchem=True, mechanism=True)
    width = model._raw_features(SMILES).shape[1]
    assert width >= 256  # at least the 256 fingerprint bits, plus the dense blocks


def test_pca_adaptation_compresses_to_the_configured_width():
    """impute -> scale -> PCA reduces the shared feature set to pca_components."""
    model = TabularFoundationModel(n_bits=256, pca_components=8)
    model._fit_reducer(SMILES)
    reduced = model._features(SMILES)
    assert reduced.shape == (len(SMILES), 8)


def test_pca_width_capped_by_available_rank():
    """When PCA applies, its width is capped by the data's rank (n_samples), never an error."""
    # pca_components below the feature count (273) triggers PCA; it is then capped by samples.
    model = TabularFoundationModel(n_bits=256, pca_components=100)
    model._fit_reducer(SMILES)
    assert model._features(SMILES).shape[1] <= len(SMILES)


def test_no_pca_when_set_already_fits_under_ceiling():
    """A feature set already within pca_components is passed through unchanged -- no PCA."""
    model = TabularFoundationModel(n_bits=256, pca_components=10_000)
    model._fit_reducer(SMILES)
    assert model._features(SMILES).shape[1] == model._raw_features(SMILES).shape[1]


def test_dense_only_feature_set_drops_the_fingerprint():
    """use_fingerprint=False gives a small dense set (no Morgan), which needs no PCA."""
    model = TabularFoundationModel(n_bits=256, use_fingerprint=False, pca_components=500)
    model._fit_reducer(SMILES)
    width = model._features(SMILES).shape[1]
    assert width == model._raw_features(SMILES).shape[1]  # passthrough, no PCA
    assert width < 256  # the 256 fingerprint bits are absent
