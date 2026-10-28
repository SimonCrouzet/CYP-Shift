"""The TDI baseline stacks a graph run's saved heads and encodings, keyed by compound."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tdi_baseline as script  # noqa: E402 -- after the scripts path insert

ENDPOINTS = [f"{e}_pIC50_direct_inhibition" for e in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")]


def _bundle(tmp_path):
    names = ["A", "B"]
    direct = pd.DataFrame({"Molecule_Name": names})
    heads = pd.DataFrame({"Molecule_Name": names})
    for endpoint in ENDPOINTS:
        direct[f"{endpoint}_pred"] = [4.0, 5.0]
    heads["CYP3A4_pIC50_direct_inhibition__tdi_drc"] = [4.5, 5.0]
    direct.to_parquet(tmp_path / "out_of_fold.parquet")
    heads.to_parquet(tmp_path / "out_of_fold_auxiliary.parquet")
    pd.DataFrame({"Molecule_Name": names, "e0": [1.0, 2.0], "e1": [3.0, 4.0]}).to_parquet(
        tmp_path / "out_of_fold_embedding.parquet"
    )
    return tmp_path


def test_the_heads_are_stacked_with_the_preincubated_shift_and_cover_the_run_s_compounds(tmp_path):
    table = pd.DataFrame({"Molecule_Name": ["B", "Z"]})  # Z is not in the run's folds
    features, covered = script.bundle_features(table, _bundle(tmp_path), embedding=False)
    assert covered.tolist() == [True, False]
    assert features.shape == (2, 6)  # four direct predictions, the head, the shift
    assert features[0, -1] == 0.0 and np.isnan(features[1]).all()  # shift = 5.0 - 5.0


def test_the_encodings_are_read_per_compound(tmp_path):
    table = pd.DataFrame({"Molecule_Name": ["B", "A"]})
    features, covered = script.bundle_features(table, _bundle(tmp_path), embedding=True)
    assert features.tolist() == [[2.0, 4.0], [1.0, 3.0]] and covered.all()
