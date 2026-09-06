"""Reading cofold features from the cache into per-row model input.

Cofolding (:mod:`cypshift.structure.cofold`) is a GPU precompute that writes one JSON
per (enzyme, compound). This side is pure pandas: it looks each row's (target, SMILES)
up in the cache and returns a fixed-width block, so the pipeline can append structure
features without Boltz or a GPU. A pair that was never cofolded contributes NaN, which a
tree handles natively and other models can impute.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.structure.cofold import DEFAULT_CACHE, STRUCTURE_FEATURE_NAMES, canonical_key

logger = logging.getLogger(__name__)


def load_pair(enzyme: str, smiles: str, cache: Path = DEFAULT_CACHE) -> dict | None:
    """Cached cofold features for one (enzyme, compound), or None if absent."""
    path = cache / enzyme / f"{canonical_key(smiles)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def structure_block(frame: pd.DataFrame, cache: Path = DEFAULT_CACHE) -> NDArray[np.float32]:
    """Per-row structure features for a frame carrying ``target`` and ``SMILES``.

    Shape ``(n_rows, len(STRUCTURE_FEATURE_NAMES))``. Rows with no cofold are NaN. The
    lookup is memoised over distinct (target, SMILES) pairs, since a frame repeats
    compounds across assays.
    """
    memo: dict[tuple[str, str], list[float]] = {}
    rows = []
    hits = 0
    for target, smiles in zip(frame["target"], frame["SMILES"], strict=True):
        key = (target, smiles)
        if key not in memo:
            found = load_pair(target, smiles, cache)
            memo[key] = (
                [float(found.get(n, np.nan)) for n in STRUCTURE_FEATURE_NAMES]
                if found
                else [np.nan] * len(STRUCTURE_FEATURE_NAMES)
            )
            hits += found is not None
        rows.append(memo[key])
    logger.info("structure cache: %d of %d distinct pairs found", hits, len(memo))
    return np.asarray(rows, dtype=np.float32)


def coverage(frame: pd.DataFrame, cache: Path = DEFAULT_CACHE) -> float:
    """Fraction of distinct (target, SMILES) pairs that have a cofold cached."""
    pairs = {(t, s) for t, s in zip(frame["target"], frame["SMILES"], strict=True)}
    if not pairs:
        return 0.0
    return sum(load_pair(t, s, cache) is not None for t, s in pairs) / len(pairs)
