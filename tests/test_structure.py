"""Tests for the Boltz-2 cofold feature extraction and cache reading.

None of these run Boltz -- they exercise the input writing, the pose/score parsing, and
the cache reader against synthetic files, so they are fast and need no GPU.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cypshift.structure.cofold import (
    STRUCTURE_FEATURE_NAMES,
    canonical_key,
    features_from_predictions,
    pose_features,
    write_boltz_input,
)
from cypshift.structure.features import coverage, load_pair, structure_block

POSE = """\
ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00  0.00           C
HETATM    2 FE   HEM B   1       0.000   0.000   0.000  1.00  0.00          FE
HETATM    3  N1  LIG C   1       2.000   0.000   0.000  1.00  0.00           N
HETATM    4  C1  LIG C   1      10.500  10.000  10.000  1.00  0.00           C
"""


def _predictions(tmp: Path, name: str = "x", pose: str = POSE, scores: bool = True) -> Path:
    pred = tmp / "predictions" / name
    pred.mkdir(parents=True)
    (pred / f"{name}_model_0.pdb").write_text(pose)
    if scores:
        (pred / f"confidence_{name}_model_0.json").write_text(
            json.dumps({"complex_plddt": 0.82, "ligand_iptm": 0.55})
        )
        (pred / f"affinity_{name}.json").write_text(
            json.dumps({"affinity_pred_value": 5.9, "affinity_probability_binary": 0.7})
        )
    return pred


def test_input_places_heme_and_ligand_and_requests_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr("cypshift.structure.cofold.sequence_for", lambda name: "MKVLLA")
    path = write_boltz_input("CYP1A2", "c1cnc[nH]1", tmp_path / "in.yaml")
    doc = yaml.safe_load(path.read_text())
    kinds = [next(iter(s)) for s in doc["sequences"]]
    assert kinds == ["protein", "ligand", "ligand"]
    ccds = [s["ligand"].get("ccd") for s in doc["sequences"] if "ligand" in s]
    assert "HEM" in ccds  # the heme cofactor is included
    assert doc["properties"][0]["affinity"]["binder"] == "C"


def test_pose_features_measure_the_coordinating_distance(tmp_path):
    (tmp_path / "m.pdb").write_text(POSE)
    features = pose_features(tmp_path / "m.pdb")
    # N1 sits 2 A from the iron; that is both the nearest ligand atom and nearest N
    assert features["ligand_iron_min_distance"] == pytest.approx(2.0)
    assert features["coordinating_n_iron_distance"] == pytest.approx(2.0)
    # C1 lies within 4.5 A of the protein CA
    assert features["ligand_protein_contacts"] == pytest.approx(1.0)


def test_pose_features_are_nan_without_an_iron(tmp_path):
    (tmp_path / "m.pdb").write_text(
        "HETATM    3  N1  LIG C   1       2.000   0.000   0.000  1.00  0.00           N\n"
    )
    features = pose_features(tmp_path / "m.pdb")
    assert np.isnan(features["ligand_iron_min_distance"])


def test_features_from_predictions_combines_pose_and_scores(tmp_path):
    pred = _predictions(tmp_path)
    features = features_from_predictions(pred, "x")
    assert set(features) == set(STRUCTURE_FEATURE_NAMES)
    assert features["affinity_pred_value"] == pytest.approx(5.9)
    assert features["complex_plddt"] == pytest.approx(0.82)
    assert features["coordinating_n_iron_distance"] == pytest.approx(2.0)


def test_missing_score_files_leave_nan(tmp_path):
    pred = _predictions(tmp_path, scores=False)
    features = features_from_predictions(pred, "x")
    assert np.isnan(features["affinity_pred_value"])
    assert features["ligand_iron_min_distance"] == pytest.approx(2.0)  # pose still parsed


def test_canonical_key_is_stable_and_short():
    assert canonical_key("CCO") == canonical_key("CCO")
    assert len(canonical_key("CCO")) == 16


def test_structure_block_returns_nan_for_uncached_pairs(tmp_path):
    (tmp_path / "CYP1A2").mkdir()
    (tmp_path / "CYP1A2" / f"{canonical_key('CCO')}.json").write_text(
        json.dumps(dict.fromkeys(STRUCTURE_FEATURE_NAMES, 0.5))
    )
    frame = pd.DataFrame({"target": ["CYP1A2", "CYP2C9"], "SMILES": ["CCO", "CCN"]})
    block = structure_block(frame, tmp_path)
    assert block.shape == (2, len(STRUCTURE_FEATURE_NAMES))
    assert np.isfinite(block[0]).all()
    assert np.isnan(block[1]).all()


def test_coverage_counts_distinct_pairs(tmp_path):
    (tmp_path / "CYP1A2").mkdir()
    (tmp_path / "CYP1A2" / f"{canonical_key('CCO')}.json").write_text(json.dumps({}))
    frame = pd.DataFrame({"target": ["CYP1A2", "CYP1A2"], "SMILES": ["CCO", "CCN"]})
    assert coverage(frame, tmp_path) == pytest.approx(0.5)


def test_load_pair_returns_none_when_absent(tmp_path):
    assert load_pair("CYP1A2", "CCO", tmp_path) is None
