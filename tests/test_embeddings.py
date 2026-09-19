"""The cached molecular-embedding feature block (reader side; no embedding model)."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.chembl import canonical
from cypshift.features.embeddings import block_for, embedding_block
from cypshift.features.molecular import featurise


def _stub_cache(tmp_path, smiles, dim=8, seed=0):
    """Write a parquet cache of canonical SMILES -> random vector, as the extractor would."""
    rng = np.random.default_rng(seed)
    canon = canonical(pd.Series(smiles))
    frame = pd.DataFrame(rng.normal(size=(len(smiles), dim)), columns=[f"e{i}" for i in range(dim)])
    frame.insert(0, "SMILES", canon.to_numpy())
    frame.to_parquet(tmp_path / "mammal.parquet")
    return tmp_path


def test_block_returns_the_cached_vector_per_compound(tmp_path):
    smiles = ["CCO", "CCN", "c1ccccc1"]
    cache = _stub_cache(tmp_path, smiles, dim=8)
    block = embedding_block("mammal", cache_dir=cache)
    out = block(["OCC", "CCN"])  # OCC == CCO canonical
    assert out.shape == (2, 8)
    # the same compound spelled differently resolves to the same cached row
    assert np.allclose(out[0], block(["CCO"])[0])


def test_a_compound_absent_from_the_cache_is_an_error(tmp_path):
    cache = _stub_cache(tmp_path, ["CCO"], dim=4)
    block = embedding_block("mammal", cache_dir=cache)
    with pytest.raises(KeyError, match="not in the mammal embedding cache"):
        block(["CCO", "CC(=O)O"])


def test_block_for_none_is_off(tmp_path):
    assert block_for("none", cache_dir=tmp_path) is None


def test_missing_cache_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="compute_embeddings"):
        embedding_block("mammal", cache_dir=tmp_path)


def test_featurise_appends_the_embedding_block(tmp_path):
    smiles = ["CCO", "CCN"]
    cache = _stub_cache(tmp_path, smiles, dim=8)
    base = featurise(smiles, n_bits=128, physchem=False, mechanism=False)
    with_emb = featurise(
        smiles, n_bits=128, physchem=False, mechanism=False,
        embedding=embedding_block("mammal", cache_dir=cache),
    )
    assert with_emb.shape[1] == base.shape[1] + 8
