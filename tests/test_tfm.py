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
    """pca_components larger than the data's rank is capped, never an error."""
    model = TabularFoundationModel(n_bits=256, pca_components=10_000)
    model._fit_reducer(SMILES)
    assert model._features(SMILES).shape[1] <= len(SMILES)
