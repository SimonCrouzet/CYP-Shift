"""The is_TDI labels as a directly trained response of the graph."""

import numpy as np
import pandas as pd
import pytest

from cypshift.data.assembly import OBSERVATION_COLUMNS, TDI_LABEL_ASSAY, tdi_label_observations
from cypshift.experiment import ExperimentConfig
from cypshift.models.graph import GraphNetwork

SMILES = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O"]


def _tdi_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Molecule_Name": [f"m{i}" for i in range(6)],
            "SMILES": SMILES,
            "CYP2D6_is_TDI": [True, False, None, False, True, False],
            "CYP3A4_is_TDI": [False, False, True, True, None, None],
        }
    )


def test_the_labels_become_one_zero_or_one_row_per_compound_and_labelled_enzyme():
    rows = tdi_label_observations(_tdi_table())
    assert list(rows.columns) == list(OBSERVATION_COLUMNS)
    assert (rows["assay"] == TDI_LABEL_ASSAY).all()
    assert rows.groupby("target").size().to_dict() == {"CYP2D6": 5, "CYP3A4": 4}
    by_enzyme = rows.groupby("target")["value"].mean().to_dict()
    assert by_enzyme == {"CYP2D6": pytest.approx(0.4), "CYP3A4": pytest.approx(0.5)}
    assert set(rows["value"]) == {0.0, 1.0}


def test_the_labels_of_compounds_without_a_direct_measurement_are_down_weighted_or_dropped():
    direct = {"m0", "m1", "m4"}
    half = tdi_label_observations(_tdi_table(), direct_names=direct, uncovered_weight=0.5)
    assert len(half) == 9
    weights = half.groupby(half["Molecule_Name"].isin(direct))["weight"].unique()
    assert weights[True].tolist() == [1.0] and weights[False].tolist() == [0.5]
    dropped = tdi_label_observations(_tdi_table(), direct_names=direct, uncovered_weight=0.0)
    assert set(dropped["Molecule_Name"]) == direct and len(dropped) == 5
    assert (
        len(tdi_label_observations(_tdi_table(), direct_names=direct)) == 9
    )  # weight 1.0 keeps all


def test_the_weight_of_uncovered_labels_needs_the_label_task():
    config = ExperimentConfig()
    config.override("model.kind", "graph")
    config.override("multifidelity.tdi_label_uncovered_weight", 0.0)
    with pytest.raises(ValueError, match="tdi_label_uncovered_weight"):
        config.validate()
    config.override("sources.tdi_label", True)
    config.validate()


def test_a_table_without_labels_gives_no_rows():
    empty = _tdi_table().assign(CYP2D6_is_TDI=None, CYP3A4_is_TDI=None)
    assert tdi_label_observations(empty).empty


def test_only_the_graph_can_train_the_label_task():
    config = ExperimentConfig()
    config.override("model.kind", "boosted")
    config.override("sources.tdi_label", True)
    with pytest.raises(ValueError, match="tdi_label"):
        config.validate()
    config.override("model.kind", "graph")
    config.validate()


def _training_rows() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    direct = []
    for i in range(24):
        value = float(rng.uniform(4, 7))
        direct.append(
            {
                "SMILES": SMILES[i % len(SMILES)],
                "target": "CYP3A4" if i % 2 else "CYP2D6",
                "assay": "direct_inhibition_drc",
                "source": "challenge",
                "data_space": "challenge",
                "value": value,
                "conf_low": value - 0.3,
                "conf_high": value + 0.3,
                "is_censored": False,
                "weight": 1.0,
            }
        )
    labels = tdi_label_observations(_tdi_table())
    return pd.concat(
        [pd.DataFrame(direct), labels.drop(columns=["Molecule_Name"])], ignore_index=True
    )


def _graph() -> GraphNetwork:
    return GraphNetwork(
        hidden=16,
        depth=2,
        ffn_hidden=16,
        epochs=1,
        batch_size=8,
        early_stopping_patience=0,
        response_assays=(TDI_LABEL_ASSAY,),
    )


def test_the_graph_calls_the_top_prevalence_share_of_its_label_scores_for_the_labelled_enzymes():
    net = _graph().fit(_training_rows())
    assert net._tdi_prevalence == {"CYP2D6": pytest.approx(0.4), "CYP3A4": pytest.approx(0.5)}
    flags = net.predict_is_time_dependent(SMILES)
    assert flags.shape == (len(SMILES), 4) and flags.dtype == bool
    # CYP1A2 and CYP2C9 have no label task, so they are never called
    assert not flags[:, 0].any() and not flags[:, 1].any()
    # six compounds, prevalence 0.4 and 0.5: at least the top 2 and 3 are called
    assert flags[:, 2].sum() >= 2 and flags[:, 3].sum() >= 3


def test_without_the_label_task_the_shift_rule_is_used(monkeypatch):
    net = GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=1, batch_size=8)
    called = []
    monkeypatch.setattr(net, "_tdi_called_from_labels", lambda smiles: called.append(smiles))
    monkeypatch.setattr(
        net, "_samples_at_assay", lambda smiles, assay: np.zeros((1, len(smiles), 4))
    )
    flags = net.predict_is_time_dependent(SMILES)
    assert not called and flags.shape == (len(SMILES), 4)
