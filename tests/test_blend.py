"""Per-endpoint blending: weight fitting, endpoint specialists, and candidate restriction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cypshift.config import REGRESSION_ENDPOINTS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from blend import blend, fit_weights, per_endpoint_best  # noqa: E402

RNG = np.random.default_rng(0)


def _member(pred_offsets: dict[str, float], n: int = 40) -> pd.DataFrame:
    """A member's out-of-fold frame: truth shared, prediction off by ``pred_offsets`` per endpoint.

    The bounds are a tight band around the truth, so soft-threshold error tracks the offset.
    """
    truth = {e: RNG.normal(5.0, 1.0, n) for e in REGRESSION_ENDPOINTS}
    frame = pd.DataFrame({"Molecule_Name": [f"m{i}" for i in range(n)]})
    for endpoint in REGRESSION_ENDPOINTS:
        frame[endpoint] = truth[endpoint]
        frame[f"{endpoint}_conf_low"] = truth[endpoint] - 0.1
        frame[f"{endpoint}_conf_high"] = truth[endpoint] + 0.1
        frame[f"{endpoint}_pred"] = truth[endpoint] + pred_offsets[endpoint]
    return frame


def test_fit_weights_equal_is_uniform():
    stack = RNG.normal(size=(20, 3))
    weights = fit_weights(stack, RNG.normal(size=20), "equal")
    assert weights == pytest.approx(np.full(3, 1 / 3))


def test_fit_weights_st_rae_concentrates_on_the_accurate_member():
    """Given one member on the truth and one far off, st_rae puts the weight on the good one."""
    truth = RNG.normal(5.0, 1.0, 60)
    stack = np.column_stack([truth, truth + 3.0])  # member 0 perfect, member 1 biased
    low, high = truth - 0.1, truth + 0.1
    weights = fit_weights(stack, truth, "st_rae", low, high)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert weights[0] > 0.9


def test_fit_weights_st_rae_needs_bounds():
    with pytest.raises(ValueError, match="bounds"):
        fit_weights(np.zeros((5, 2)), np.zeros(5), "st_rae")


def test_per_endpoint_best_picks_the_specialist_for_each_endpoint():
    """Two members, each accurate on two endpoints; the best per endpoint is the accurate one."""
    a = _member({REGRESSION_ENDPOINTS[0]: 0.0, REGRESSION_ENDPOINTS[1]: 3.0,
                 REGRESSION_ENDPOINTS[2]: 0.0, REGRESSION_ENDPOINTS[3]: 3.0})
    b = _member({REGRESSION_ENDPOINTS[0]: 3.0, REGRESSION_ENDPOINTS[1]: 0.0,
                 REGRESSION_ENDPOINTS[2]: 3.0, REGRESSION_ENDPOINTS[3]: 0.0})
    best = per_endpoint_best({"A": a, "B": b})
    assert best[REGRESSION_ENDPOINTS[0].split("_")[0]] == "A"
    assert best[REGRESSION_ENDPOINTS[1].split("_")[0]] == "B"
    assert best[REGRESSION_ENDPOINTS[2].split("_")[0]] == "A"
    assert best[REGRESSION_ENDPOINTS[3].split("_")[0]] == "B"


def test_endpoint_candidates_restrict_which_member_serves_an_endpoint():
    """Restricting an endpoint to one member makes the blend equal that member's prediction."""
    a = _member(dict.fromkeys(REGRESSION_ENDPOINTS, 0.0))
    b = _member(dict.fromkeys(REGRESSION_ENDPOINTS, 2.0))
    endpoint = REGRESSION_ENDPOINTS[0]
    enzyme = endpoint.split("_")[0]
    predictions, _, weights = blend(
        {"A": a, "B": b}, "equal", endpoint_candidates={enzyme: ["B"]}
    )
    assert set(weights[enzyme]) == {"B"}
    expected = b.set_index("Molecule_Name").loc[predictions["Molecule_Name"], f"{endpoint}_pred"]
    assert np.allclose(predictions[endpoint].to_numpy(), expected.to_numpy())


def test_fit_mask_restricts_the_rows_the_weights_are_fit_on():
    """With a fit mask, every compound still gets a blended prediction (scoring is the caller's)."""
    a = _member(dict.fromkeys(REGRESSION_ENDPOINTS, 0.0))
    b = _member(dict.fromkeys(REGRESSION_ENDPOINTS, 2.0))
    mask = np.zeros(len(a), dtype=bool)
    mask[: len(a) // 2] = True
    predictions, truth, _ = blend({"A": a, "B": b}, "st_rae", fit_mask=mask)
    assert len(predictions) == len(a)
    assert predictions[list(REGRESSION_ENDPOINTS)].to_numpy().shape == (len(a), 4)
