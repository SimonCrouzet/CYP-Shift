"""The best-model spec: its recipes validate, the blend is the weighted mean, and the build
standardises the blend to the calibration targets."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import best_model  # noqa: E402 -- after the scripts path insert

from cypshift.config import REGRESSION_ENDPOINTS  # noqa: E402

ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")


def _member_file(path: Path, shift: float, seed: int, order=None) -> str:
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame({"SMILES": ["C", "CC", "CCC", "CCCC"], "Molecule_Name": list("abcd")})
    for column in REGRESSION_ENDPOINTS:
        frame[column] = generator.normal(5.0 + shift, 1.0, size=4)
    if order is not None:
        frame = frame.iloc[order].reset_index(drop=True)
    frame.to_csv(path, index=False)
    return str(path)


def _spec(tmp_path: Path, weights=(0.75, 0.25), order=None) -> dict:
    targets = {e: {"mean": 4.0, "sd": 0.5} for e in ENZYMES}
    (tmp_path / "targets.json").write_text(json.dumps(targets))
    return {
        "name": "toy",
        "members": {
            "a": {
                "weight": weights[0],
                "file": _member_file(tmp_path / "a.csv", 0.0, 0),
                "recipe": {},
            },
            "b": {
                "weight": weights[1],
                "file": _member_file(tmp_path / "b.csv", 1.0, 1, order),
                "recipe": {},
            },
        },
        "calibration": {"targets": str(tmp_path / "targets.json")},
    }


def test_the_committed_spec_has_valid_recipes_and_weights_summing_to_one():
    spec = best_model.load_spec(Path(__file__).resolve().parents[1] / best_model.SPEC)
    best_model.check(spec)
    assert set(spec["members"]) >= {"graph_rft"}


def test_weights_must_sum_to_one(tmp_path):
    spec = _spec(tmp_path, weights=(0.5, 0.2))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="sum to 1"):
        best_model.load_spec(path)


def test_the_blend_is_the_weighted_mean_of_the_member_files(tmp_path):
    spec = _spec(tmp_path)
    a = pd.read_csv(spec["members"]["a"]["file"])
    b = pd.read_csv(spec["members"]["b"]["file"])
    blended = best_model.blend(spec)
    for column in REGRESSION_ENDPOINTS:
        np.testing.assert_allclose(blended[column], 0.75 * a[column] + 0.25 * b[column])


def test_a_member_listing_the_compounds_in_another_order_is_refused(tmp_path):
    spec = _spec(tmp_path, order=[3, 2, 1, 0])
    with pytest.raises(ValueError, match="another order"):
        best_model.blend(spec)


def test_build_standardises_each_endpoint_to_the_targets_and_records_provenance(tmp_path):
    spec = _spec(tmp_path)
    output = best_model.build(spec, tmp_path / "out" / "candidate.csv")
    built = pd.read_csv(output)
    for enzyme in ENZYMES:
        column = f"{enzyme}_pIC50_direct_inhibition"
        assert built[column].mean() == pytest.approx(4.0)
        assert built[column].std(ddof=0) == pytest.approx(0.5)
    provenance = json.loads((tmp_path / "out" / "candidate.provenance.json").read_text())
    assert set(provenance["members"]) == {"a", "b"} and provenance["spread"] == 1.0
    assert (tmp_path / "out" / "candidate_uncalibrated.csv").exists()


def test_the_recipe_file_carries_the_member_name_and_its_settings():
    spec = {"members": {"m": {"recipe": {"model.kind": "boosted"}}}}
    assert best_model.recipe(spec, "m") == {"name": "m", "model.kind": "boosted"}
