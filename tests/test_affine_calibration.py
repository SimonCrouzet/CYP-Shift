"""The affine recalibration map fit by scripts/calibrate.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import calibrate  # noqa: E402
from calibrate import affine_for_endpoint, load_out_of_fold  # noqa: E402

from cypshift.config import REGRESSION_ENDPOINTS  # noqa: E402


def _compressed(n=4000):
    """Truth, and a prediction that ranks it well but under-spreads it."""
    rng = np.random.default_rng(0)
    truth = rng.normal(5.0, 1.2, n)
    pred = 0.5 * truth + rng.normal(0.0, 0.8, n)
    band = np.full(n, 0.3)
    return truth, pred, truth - band, truth + band


def test_variance_match_rho_targets_the_r2_optimal_spread():
    """variance_match_rho sets the slope to rho * sigma_true / sigma_pred."""
    truth, pred, low, high = _compressed()
    slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match_rho", 0.5)
    rho = np.corrcoef(truth, pred)[0, 1]
    assert slope == pytest.approx(rho * truth.std() / pred.std(), rel=1e-6)


def test_rho_scaling_is_less_aggressive_than_full_variance_match():
    """Because rho < 1, de-shrinking to rho*sigma never overshoots the full-sigma target."""
    truth, pred, low, high = _compressed()
    rho_slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match_rho", 0.5)
    full_slope, _ = affine_for_endpoint(truth, pred, low, high, "variance_match", 0.5)
    assert 0.0 < rho_slope < full_slope


def test_offset_leaves_the_slope_at_one():
    """The offset method shifts without rescaling, so the fitted slope is exactly one."""
    truth, pred, low, high = _compressed()
    slope, _ = affine_for_endpoint(truth, pred, low, high, "offset", 0.5)
    assert slope == 1.0


def test_affine_minimises_interval_error_and_need_not_expand():
    """The affine map lowers ST-RAE, and -- the note's point -- its optimal slope is free to
    sit below one rather than being forced to de-shrink."""
    from cypshift.eval.metrics import soft_threshold_rae

    truth, pred, low, high = _compressed()
    slope, intercept = affine_for_endpoint(truth, pred, low, high, "affine", 0.5)
    before = soft_threshold_rae(truth, pred, low, high)
    after = soft_threshold_rae(truth, slope * pred + intercept, low, high)
    assert after < before
    assert slope != pytest.approx(1.0)  # it adapts; here the narrow bands make it shrink


def test_affine_slope_is_clamped_to_the_bound():
    """The restrained affine cannot run past its slope bound even when the fit wants more.

    The unclamped optimum on this fixture is about 0.72; a (0.25, 0.5) band forces it to 0.5.
    """
    truth, pred, low, high = _compressed()
    slope, _ = affine_for_endpoint(truth, pred, low, high, "affine", 0.5, slope_bounds=(0.25, 0.5))
    assert slope == pytest.approx(0.5)


def _oof_frame(pred_value: float, n: int = 12) -> "object":
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"Molecule_Name": [f"c{i}" for i in range(n)]})
    for endpoint in REGRESSION_ENDPOINTS:
        frame[endpoint] = np.linspace(4.0, 7.0, n)
        frame[f"{endpoint}_conf_low"] = frame[endpoint] - 0.2
        frame[f"{endpoint}_conf_high"] = frame[endpoint] + 0.2
        frame[f"{endpoint}_pred"] = pred_value
    return frame


def test_load_out_of_fold_averages_member_predictions(tmp_path, monkeypatch):
    """Two members' out-of-fold predictions are averaged row for row into the ensemble OOF."""
    monkeypatch.setattr(calibrate, "BUNDLE_ROOT", tmp_path)
    for member, value in (("m1", 1.0), ("m2", 3.0)):
        bundle = tmp_path / "rev0" / f"reg_{member}_seed0"
        bundle.mkdir(parents=True)
        _oof_frame(value).to_parquet(bundle / "out_of_fold.parquet", index=False)
    averaged = load_out_of_fold("rev0", ["m1", "m2"], 0)
    for endpoint in REGRESSION_ENDPOINTS:
        assert averaged[f"{endpoint}_pred"].to_numpy() == pytest.approx(2.0)
