"""Reading precomputed molecular foundation-model embeddings as a feature block.

The embedding model (e.g. MAMMAL, IBM's biomed multi-modal model) is heavy and pulls a
different dependency stack, so it is never imported here. Instead it runs offline --
``scripts/compute_embeddings.py`` in its own environment -- and writes one vector per
compound to a cache under ``data/external/embeddings/<source>.parquet`` keyed by canonical
SMILES. This module only reads that cache and hands the vectors to
:func:`cypshift.features.molecular.featurise` through its ``embedding`` hook, so the main
environment stays free of the embedding model. A compound absent from the cache is an
error, not a zero-fill: it means the cache is stale and must be regenerated, and a silent
zero would place the compound at a real point in embedding space.
"""

import logging
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.data.chembl import canonical

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/external/embeddings")


@cache
def _load(source: str, cache_dir: Path) -> tuple[dict[str, int], NDArray[np.float32]]:
    """The embedding table for one source: canonical SMILES -> row, and the matrix.

    Cached in-process because a fit re-featurises the same compounds many times.

    Raises:
        FileNotFoundError: if the source has no cache (run scripts/compute_embeddings.py).
    """
    path = cache_dir / f"{source}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no {source} embedding cache at {path}; "
            f"run scripts/compute_embeddings.py --source {source}"
        )
    frame = pd.read_parquet(path)
    vectors = frame.drop(columns="SMILES").to_numpy(dtype=np.float32)
    index = {smiles: row for row, smiles in enumerate(frame["SMILES"])}
    logger.info("loaded %d %s embeddings of width %d", len(index), source, vectors.shape[1])
    return index, vectors


def embedding_block(
    source: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Callable[[Sequence[str]], NDArray[np.float32]]:
    """A ``featurise``-compatible callable returning the cached embedding per compound.

    Matching is on canonical SMILES, since the cache and the query reach the same compound
    under different spellings.
    """
    index, vectors = _load(source, cache_dir)

    def block(smiles: Sequence[str]) -> NDArray[np.float32]:
        rows = canonical(pd.Series(list(smiles)))
        missing = [s for s, c in zip(smiles, rows, strict=True) if c not in index]
        if missing:
            raise KeyError(
                f"{len(missing)} compounds are not in the {source} embedding cache "
                f"(e.g. {missing[0]}); regenerate it with scripts/compute_embeddings.py"
            )
        return vectors[[index[c] for c in rows]]

    return block


def block_for(
    source: str, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Callable[[Sequence[str]], NDArray[np.float32]] | None:
    """The embedding block for a source, or None when embeddings are off (``none``)."""
    if source == "none":
        return None
    return embedding_block(source, cache_dir)
