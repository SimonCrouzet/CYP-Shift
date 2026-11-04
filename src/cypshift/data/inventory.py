"""Per-group summary of the auxiliary training rows.

One row per (target, assay, source, data space) group: observation and distinct-compound counts,
total training weight, value quantiles and censored fraction (value units differ between assays,
so groups are never pooled), and Butina cluster statistics on the Morgan fingerprints.
"""

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["target", "assay", "source", "data_space"]
#: Groups above this many rows are not clustered (the distance matrix grows with the square).
MAX_CLUSTERED_ROWS = 8000


def cluster_labels(smiles: list[str], distance: float) -> np.ndarray:
    """Butina cluster index of each SMILES; two compounds share a cluster only through a centroid
    within Tanimoto ``distance`` (1 - similarity) of both."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    from cypshift.data.splits import morgan_fingerprints

    fingerprints = morgan_fingerprints(smiles)
    distances: list[float] = []
    for row in range(1, len(fingerprints)):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[row], fingerprints[:row])
        distances.extend(1.0 - np.asarray(similarity))
    labels = np.empty(len(smiles), dtype=int)
    for index, members in enumerate(Butina.ClusterData(distances, len(smiles), distance, True)):
        labels[list(members)] = index
    return labels


def group_inventory(group: pd.DataFrame, cluster_distance: float) -> dict:
    from cypshift.data.chembl import canonical

    weight = group["weight"].to_numpy(float)
    value = group["value"].to_numpy(float)
    summary = {
        "rows": len(group),
        "compounds": int(canonical(group["SMILES"]).nunique()),
        "weight": float(weight.sum()),
        "value_q10": float(np.quantile(value, 0.1)),
        "value_q50": float(np.quantile(value, 0.5)),
        "value_q90": float(np.quantile(value, 0.9)),
        "censored": float(group["is_censored"].astype(bool).mean()),
        "clusters": np.nan,
        "largest_cluster_rows": np.nan,
        "top10_cluster_weight": np.nan,
    }
    if len(group) > MAX_CLUSTERED_ROWS:
        return summary
    labels = cluster_labels(group["SMILES"].tolist(), cluster_distance)
    sizes = np.bincount(labels)
    by_weight = np.bincount(labels, weights=weight)
    summary["clusters"] = int(sizes.size)
    summary["largest_cluster_rows"] = float(sizes.max() / len(group))
    summary["top10_cluster_weight"] = float(np.sort(by_weight)[::-1][:10].sum() / weight.sum())
    return summary


def auxiliary_inventory(table: pd.DataFrame, cluster_distance: float = 0.6) -> pd.DataFrame:
    """The per-group summary of an auxiliary table (columns as in ``pipeline.auxiliary_frames``)."""
    rows = []
    for key, group in table.groupby(GROUP_COLUMNS, sort=True):
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, key, strict=True)),
                **group_inventory(group, cluster_distance),
            }
        )
    return pd.DataFrame(rows)
