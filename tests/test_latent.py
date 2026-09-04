"""The joint model of direct potency and the preincubation shift."""

import numpy as np
import pandas as pd
import pytest
import torch

from cypshift.config import ISOFORMS, TDI_SHIFT_THRESHOLD
from cypshift.data.assembly import build_observations
from cypshift.models.latent import LatentPotencyModel, observation_loss

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]]


def _t(*values):
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def test_a_direct_row_constrains_the_direct_value_only():
    direct, shift, low, high, pre, w = _t([5.0], [2.0], [4.8], [5.2], [0.0], [1.0])
    assert float(observation_loss(direct, shift, low, high, pre, w)) == 0.0


def test_a_preincubated_row_constrains_direct_plus_shift():
    # direct 5.0 with shift 0.5 predicts 5.5, which sits inside [5.3, 5.7].
    direct, shift, low, high, pre, w = _t([5.0], [0.5], [5.3], [5.7], [1.0], [1.0])
    assert float(observation_loss(direct, shift, low, high, pre, w)) == 0.0
    # The same row with no shift predicts 5.0 and is charged the distance to 5.3.
    direct, shift, low, high, pre, w = _t([5.0], [0.0], [5.3], [5.7], [1.0], [1.0])
    assert float(observation_loss(direct, shift, low, high, pre, w)) == pytest.approx(0.3)


def test_the_shift_receives_gradient_only_from_preincubated_rows():
    shift = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
    direct, low, high, w = _t([9.0], [4.8], [5.2], [1.0])
    observation_loss(direct, shift, low, high, torch.tensor([0.0]), w).backward()
    assert float(shift.grad) == 0.0


def _observations(seed=0, with_preincubated=True):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        y = rng.uniform(3.5, 7.0, len(SMILES))
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    inhibition = pd.DataFrame(frame)
    tdi = None
    if with_preincubated:
        tdi = inhibition[["Molecule_Name", "SMILES"]].copy()
        for isoform in ISOFORMS:
            tdi[f"{isoform}_pIC50_TDI_condition"] = (
                inhibition[f"{isoform}_pIC50_direct_inhibition"] + 0.4
            )
    return build_observations(inhibition, tdi=tdi)


def _model(**kwargs):
    settings = {"n_bits": 128, "hidden": 32, "epochs": 25, "n_forward_passes": 4}
    return LatentPotencyModel(**{**settings, **kwargs})


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="model is not fitted"):
        _model().predict_direct(SMILES[:3])


def test_fitting_on_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="no finite measurements"):
        _model().fit(_observations().iloc[:0])


def test_every_requested_output_derives_from_one_fit():
    model = _model().fit(_observations())
    smiles = SMILES[:6]
    assert model.predict_direct(smiles).shape == (6, len(ISOFORMS))
    assert model.predict_shift(smiles).shape == (6, len(ISOFORMS))
    assert model.predict_preincubated(smiles).shape == (6, len(ISOFORMS))
    assert model.predict_is_time_dependent(smiles).shape == (6, len(ISOFORMS))


def test_the_preincubated_prediction_is_direct_plus_shift():
    model = _model().fit(_observations())
    smiles = SMILES[:5]
    np.testing.assert_allclose(
        model.predict_preincubated(smiles),
        model.predict_direct(smiles) + model.predict_shift(smiles),
        rtol=1e-4,
    )


def test_a_learned_positive_shift_is_recovered():
    """The synthetic data has every preincubated value 0.4 above its direct value."""
    model = _model(epochs=120, shift_penalty=0.0).fit(_observations())
    assert model.predict_shift(SMILES).mean() > 0.1


def test_the_time_dependence_call_follows_the_published_rule():
    model = _model().fit(_observations())
    smiles = SMILES[:8]
    direct = model.predict_direct(smiles)
    shift = model.predict_shift(smiles)
    expected = np.where(
        direct > 4.0, shift > TDI_SHIFT_THRESHOLD, direct + shift > 4.0 + TDI_SHIFT_THRESHOLD
    )
    np.testing.assert_array_equal(model.predict_is_time_dependent(smiles), expected)


def test_training_without_any_preincubated_row_still_fits():
    model = _model().fit(_observations(with_preincubated=False))
    assert np.isfinite(model.predict_direct(SMILES[:3])).all()
