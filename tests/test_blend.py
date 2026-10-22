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
N = 40
#: One truth table shared by every member, keyed to the shared molecule IDs -- members must
#: differ only in their predictions, so a per-endpoint comparison is apples-to-apples.
SHARED_TRUTH = {e: np.random.default_rng(0).normal(5.0, 1.0, N) for e in REGRESSION_ENDPOINTS}
SHARED_NAMES = [f"m{i}" for i in range(N)]


def _member(pred_offsets: dict[str, float]) -> pd.DataFrame:
    """A member's out-of-fold frame over the shared truth: prediction off by ``pred_offsets``.

    The bounds are a tight band around the truth, so soft-threshold error tracks the offset.
    """
    frame = pd.DataFrame({"Molecule_Name": SHARED_NAMES})
    for endpoint in REGRESSION_ENDPOINTS:
        truth = SHARED_TRUTH[endpoint]
        frame[endpoint] = truth
        frame[f"{endpoint}_conf_low"] = truth - 0.1
        frame[f"{endpoint}_conf_high"] = truth + 0.1
        frame[f"{endpoint}_pred"] = truth + pred_offsets[endpoint]
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
    a = _member(
        {
            REGRESSION_ENDPOINTS[0]: 0.0,
            REGRESSION_ENDPOINTS[1]: 3.0,
            REGRESSION_ENDPOINTS[2]: 0.0,
            REGRESSION_ENDPOINTS[3]: 3.0,
        }
    )
    b = _member(
        {
            REGRESSION_ENDPOINTS[0]: 3.0,
            REGRESSION_ENDPOINTS[1]: 0.0,
            REGRESSION_ENDPOINTS[2]: 3.0,
            REGRESSION_ENDPOINTS[3]: 0.0,
        }
    )
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
    predictions, _, weights = blend({"A": a, "B": b}, "equal", endpoint_candidates={enzyme: ["B"]})
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


def _split_member(good_first: bool) -> pd.DataFrame:
    """Matches the truth on the first half of the rows when ``good_first``, the second half else."""
    half = N // 2
    frame = pd.DataFrame({"Molecule_Name": SHARED_NAMES})
    for endpoint in REGRESSION_ENDPOINTS:
        truth = SHARED_TRUTH[endpoint]
        frame[endpoint] = truth
        frame[f"{endpoint}_conf_low"] = truth - 0.1
        frame[f"{endpoint}_conf_high"] = truth + 0.1
        pred = truth.copy()
        pred[half:] += 5.0 if good_first else 0.0
        pred[:half] += 0.0 if good_first else 5.0
        frame[f"{endpoint}_pred"] = pred
    return frame


def test_fit_mask_fits_weights_on_the_fitting_rows_only():
    """The weights must reflect only the fitting rows: a member accurate on the fit half but
    wrong on the held half should still win the weight."""
    fit_winner = _split_member(good_first=True)  # accurate on the first (fit) half
    held_winner = _split_member(good_first=False)  # accurate only on the held half
    mask = np.zeros(N, dtype=bool)
    mask[: N // 2] = True  # fit = first half
    _, _, weights = blend({"fit": fit_winner, "held": held_winner}, "st_rae", fit_mask=mask)
    for endpoint in REGRESSION_ENDPOINTS:
        w = weights[endpoint.split("_")[0]]
        assert w["fit"] > w["held"]
