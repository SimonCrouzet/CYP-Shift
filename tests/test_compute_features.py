"""The feature cache script computes only the compounds a block's cache lacks when updating."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compute_features as script  # noqa: E402 -- after the scripts path insert

from cypshift.features import molecular  # noqa: E402


def _run(monkeypatch, tmp_path, smiles, *flags):
    monkeypatch.setattr(molecular, "FEATURE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(script, "gather_smiles", lambda *args, **kwargs: list(smiles))
    monkeypatch.setattr(script, "gather_aux_smiles", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["compute_features.py", "--bits", "64", "--jobs", "1", *flags])
    script.main()


def _stored(tmp_path, name):
    with np.load(tmp_path / f"{name}.npz", allow_pickle=False) as stored:
        return stored["smiles"].tolist(), stored["values"], stored["columns"].tolist()


def test_update_computes_only_the_new_compounds_and_keeps_the_old_rows(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, ["CCO", "c1ccccc1"])
    before, values, columns = _stored(tmp_path, "physchem")
    assert before == ["CCO", "c1ccccc1"]
    values[0, 0] = 777.0  # a marker: an untouched row survives an update
    np.savez_compressed(
        tmp_path / "physchem.npz",
        smiles=np.array(before),
        values=values,
        columns=np.array(columns),
    )
    _run(monkeypatch, tmp_path, ["CCO", "c1ccccc1", "CC(=O)O"], "--update")
    after, updated, _ = _stored(tmp_path, "physchem")
    assert after == ["CCO", "c1ccccc1", "CC(=O)O"]
    assert updated[0, 0] == 777.0  # the cached row was not recomputed
    expected = molecular.compute_physicochemical(["CC(=O)O"]).to_numpy(np.float32)
    np.testing.assert_allclose(updated[2], expected[0])
    morgan, counts, _ = _stored_morgan(tmp_path)
    assert morgan == after and counts.shape == (3, 64)


def _stored_morgan(tmp_path):
    with np.load(tmp_path / "morgan_r2_b64.npz", allow_pickle=False) as stored:
        return stored["smiles"].tolist(), stored["values"], None
