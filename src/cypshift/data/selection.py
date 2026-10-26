"""Reduce a large auxiliary group to a fixed number of its rows.

A group is one (target, assay, source, data space) block of the auxiliary table. Groups at or
under the cap are untouched; larger ones keep ``max_rows`` rows chosen by one of:

* ``clusters`` -- k-means with ``k = max_rows`` on a 128-component SVD projection of the Morgan
  bit vectors; the row nearest each cluster centre is kept;
* ``maxmin`` -- RDKit's MaxMin picker on Tanimoto distance;
* ``nearest`` -- the rows with the highest Tanimoto similarity to any reference compound.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SELECTIONS = ("none", "clusters", "maxmin", "nearest")
GROUP_COLUMNS = ["target", "assay", "source", "data_space"]
#: Dimension of the SVD projection of the bit vectors that k-means runs in.
SVD_COMPONENTS = 128


def _dense(fingerprints: Sequence) -> NDArray[np.float32]:
    """The bit vectors as a (rows, bits) 0/1 matrix."""
    rows = [
        np.frombuffer(f.ToBitString().encode(), dtype=np.uint8) - ord("0") for f in fingerprints
    ]
    return np.stack(rows).astype(np.float32)


def _clusters(fingerprints: Sequence, k: int, seed: int) -> NDArray[np.intp]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    matrix = _dense(fingerprints)
    components = min(SVD_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1] - 1)
    reduced = TruncatedSVD(components, random_state=seed).fit_transform(matrix)
    model = MiniBatchKMeans(n_clusters=k, n_init=1, random_state=seed).fit(reduced)
    distance = np.linalg.norm(reduced - model.cluster_centers_[model.labels_], axis=1)
    return pd.Series(distance).groupby(model.labels_).idxmin().to_numpy(np.intp)


def _maxmin(fingerprints: Sequence, k: int, seed: int) -> NDArray[np.intp]:
    from rdkit.SimDivFilters import rdSimDivPickers

    picks = rdSimDivPickers.MaxMinPicker().LazyBitVectorPick(
        list(fingerprints), len(fingerprints), k, seed=seed
    )
    return np.array(list(picks), dtype=np.intp)


def _nearest(fingerprints: Sequence, k: int, reference: Sequence) -> NDArray[np.intp]:
    from cypshift.data.splits import max_similarity_to

    similarity = max_similarity_to(fingerprints, reference)
    return np.argsort(-similarity, kind="stable")[:k].astype(np.intp)


def select_rows(
    smiles: Sequence[str],
    k: int,
    method: str,
    reference_smiles: Sequence[str] = (),
    seed: int = 0,
) -> NDArray[np.intp]:
    """Positions of the ``k`` rows of ``smiles`` that ``method`` keeps (fewer if clusters merge)."""
    from cypshift.data.splits import morgan_fingerprints

    fingerprints = morgan_fingerprints(list(smiles))
    if method == "clusters":
        return _clusters(fingerprints, k, seed)
    if method == "maxmin":
        return _maxmin(fingerprints, k, seed)
    if method == "nearest":
        if len(reference_smiles) == 0:
            raise ValueError("method 'nearest' needs reference compounds")
        return _nearest(fingerprints, k, morgan_fingerprints(list(reference_smiles)))
    raise ValueError(f"method={method!r} is not one of {SELECTIONS[1:]}")


def reduce_groups(
    table: pd.DataFrame,
    max_rows: int,
    method: str,
    reference_smiles: Sequence[str] = (),
    seed: int = 0,
) -> pd.DataFrame:
    """``table`` with every group of more than ``max_rows`` rows reduced to ``max_rows``."""
    kept = []
    for key, group in table.groupby(GROUP_COLUMNS, sort=False):
        if len(group) <= max_rows:
            kept.append(group.index.to_numpy())
            continue
        chosen = select_rows(group["SMILES"].tolist(), max_rows, method, reference_smiles, seed)
        kept.append(group.index.to_numpy()[chosen])
        logger.info(
            "%s: kept %d of %d rows (%s)", "/".join(map(str, key)), len(chosen), len(group), method
        )
    return table.loc[table.index.isin(np.concatenate(kept))]
