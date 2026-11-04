"""Per-group inventory of the auxiliary rows."""

import pandas as pd

from cypshift.data.inventory import auxiliary_inventory, cluster_labels

NEAR = ["CCCCCCO", "CCCCCCCO", "CCCCCO"]
FAR = ["c1ccc2ccccc2c1", "C1CCOC1C(=O)O"]


def test_close_compounds_share_a_cluster_and_distant_ones_do_not():
    labels = cluster_labels(NEAR + FAR, 0.6)
    assert len(set(labels[:3])) == 1 and len(set(labels)) >= 3


def test_the_inventory_counts_rows_compounds_weight_and_clusters_per_group():
    rows = pd.DataFrame(
        {
            "SMILES": NEAR + FAR,
            "target": "CYP2D6",
            "assay": "luminescent_ac50",
            "source": "_pooled",
            "data_space": "near_aux",
            "value": [4.0, 5.0, 6.0, 4.5, 5.5],
            "weight": [1.0, 1.0, 1.0, 2.0, 3.0],
            "is_censored": [False, False, False, False, True],
        }
    )
    other = rows.iloc[:2].assign(assay="reversible_pic50")
    inventory = auxiliary_inventory(pd.concat([rows, other], ignore_index=True)).set_index("assay")
    ac50 = inventory.loc["luminescent_ac50"]
    assert (ac50["rows"], ac50["compounds"], ac50["weight"]) == (5, 5, 8.0)
    assert ac50["censored"] == 0.2 and ac50["value_q50"] == 5.0
    assert ac50["largest_cluster_rows"] == 0.6 and ac50["clusters"] == 3
    assert inventory.loc["reversible_pic50", "rows"] == 2
