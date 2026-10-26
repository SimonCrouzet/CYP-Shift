"""The probe solver recovers a known test-set mean, spread and correlation from the probe R²."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibration_probe as probe  # noqa: E402 -- after the scripts path insert

ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
#: (true mean, true sd, correlation of prediction and truth, prediction mean, prediction sd)
TRUTH = {
    "CYP1A2": (5.6, 0.9, 0.70, 5.1, 0.55),
    "CYP2C9": (4.8, 0.8, 0.76, 4.9, 0.60),
    "CYP2D6": (4.2, 0.6, 0.45, 4.7, 0.40),
    "CYP3A4": (4.9, 1.0, 0.82, 4.8, 0.80),
}
SHIFT, SCALE = 0.5, 1.5


def _world(n: int = 750, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    base = pd.DataFrame({"SMILES": ["C"] * n, "Molecule_Name": [f"m{i}" for i in range(n)]})
    truth = base.copy()
    for enzyme, (mean_y, sd_y, r, mean_p, sd_p) in TRUTH.items():
        z_y = generator.normal(size=n)
        z_p = r * z_y + np.sqrt(1 - r**2) * generator.normal(size=n)
        truth[probe._column(enzyme)] = mean_y + sd_y * z_y
        base[probe._column(enzyme)] = probe.standardise_to(z_p, mean_p, sd_p)
    return base, truth


def _metrics(base_frame, truth, paths) -> pd.DataFrame:
    scored = {"base": base_frame, "a": pd.read_csv(paths[0]), "b": pd.read_csv(paths[1])}
    rows = []
    for enzyme in ENZYMES:
        column = probe._column(enzyme)
        for name, frame in scored.items():
            kind = "base" if name == "base" else probe.PROBE_PLAN[name][enzyme]
            rows.append(
                {
                    "endpoint": column,
                    "probe": kind,
                    "r2": r2_score(truth[column], frame[column]),
                }
            )
    return pd.DataFrame(rows)


def test_probe_files_change_one_transformation_per_endpoint_and_nothing_else(tmp_path):
    base, _ = _world()
    base_path = tmp_path / "base.csv"
    base.to_csv(base_path, index=False)
    paths = probe.write_probes(base_path, tmp_path, SHIFT, SCALE)
    assert [p.name for p in paths] == ["base_probe_a.csv", "base_probe_b.csv"]
    for name, path in zip("ab", paths, strict=True):
        frame = pd.read_csv(path)
        assert frame[["SMILES", "Molecule_Name"]].equals(base[["SMILES", "Molecule_Name"]])
        for enzyme, kind in probe.PROBE_PLAN[name].items():
            column = probe._column(enzyme)
            before, after = base[column], frame[column]
            if kind == "shift":
                assert np.allclose(after - before, SHIFT)
            else:
                assert after.mean() == pytest.approx(before.mean())
                assert after.std() == pytest.approx(SCALE * before.std())
    # the two files are complementary: every endpoint gets each transformation once
    assert all(
        {probe.PROBE_PLAN["a"][e], probe.PROBE_PLAN["b"][e]} == {"shift", "scale"} for e in ENZYMES
    )


def test_solve_recovers_the_true_moments_and_the_optimal_target(tmp_path):
    base, truth = _world()
    base_path = tmp_path / "base.csv"
    base.to_csv(base_path, index=False)
    paths = probe.write_probes(base_path, tmp_path, SHIFT, SCALE)
    _metrics(base, truth, paths).to_csv(tmp_path / "metrics.csv", index=False)
    targets = probe.solve(base_path, tmp_path / "metrics.csv", SHIFT, SCALE, noise=0.01)
    assert set(targets) == set(ENZYMES)
    for enzyme, (_, _, r, _, _) in TRUTH.items():
        found = targets[enzyme]
        y = truth[probe._column(enzyme)]
        assert found["mean_true"] == pytest.approx(y.mean(), abs=1e-6)
        assert found["sd_true"] == pytest.approx(y.std(ddof=0), abs=1e-6)
        assert found["mean"] == pytest.approx(y.mean(), abs=1e-6)
        assert found["sd"] == pytest.approx(found["r"] * y.std(ddof=0))
        assert found["r"] == pytest.approx(r, abs=0.06)  # the sample correlation, not the draw's r
        low, high = found["mean_true_10_90"]
        assert low <= found["mean_true"] <= high


def test_apply_standardises_each_endpoint_to_its_target(tmp_path):
    base, _ = _world()
    path = tmp_path / "sub.csv"
    base.to_csv(path, index=False)
    targets = {e: {"mean": 4.0 + i, "sd": 0.5 + i / 10} for i, e in enumerate(ENZYMES)}
    out = tmp_path / "out.csv"
    probe.apply_targets(path, targets, out, only=["CYP2C9", "CYP3A4"])
    result = pd.read_csv(out)
    for e, target in targets.items():
        column = probe._column(e)
        if e in ("CYP2C9", "CYP3A4"):
            assert result[column].mean() == pytest.approx(target["mean"])
            assert result[column].std(ddof=0) == pytest.approx(target["sd"])
            assert np.corrcoef(result[column], base[column])[0, 1] == pytest.approx(1.0)
        else:
            assert np.allclose(result[column], base[column])


def test_apply_clips_to_the_target_bounds(tmp_path):
    base, _ = _world()
    path = tmp_path / "sub.csv"
    base.to_csv(path, index=False)
    targets = {"CYP2D6": {"mean": 3.0, "sd": 1.5, "min": 2.0, "max": 5.0}}
    out = tmp_path / "out.csv"
    probe.apply_targets(path, targets, out, only=None)
    values = pd.read_csv(out)[probe._column("CYP2D6")]
    assert values.min() == 2.0 and values.max() == 5.0
