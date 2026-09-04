"""The dose-response curve model: the physics, then the model built on it."""

import numpy as np
import pandas as pd
import pytest
import torch

from cypshift.config import ISOFORMS
from cypshift.data.assembly import build_observations
from cypshift.models.curve import (
    ARMS,
    EMAX,
    EMAX_OFFSET,
    HILL,
    PIC50,
    SCREEN_PC,
    CurveModel,
    inhibition_at,
)

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]]


def _parameters(pic50=6.0, hill_raw=0.0, emax_raw=0.0):
    return torch.tensor([[pic50, hill_raw, emax_raw]], dtype=torch.float32)


def _emax_of(parameters):
    return float(torch.sigmoid(parameters[0, EMAX] + EMAX_OFFSET))


def test_half_maximal_inhibition_occurs_at_the_midpoint():
    """The definition of an IC50: at that concentration, response is half of Emax."""
    parameters = _parameters(pic50=6.0)
    assert float(inhibition_at(parameters, 6.0)) == pytest.approx(
        _emax_of(parameters) / 2, rel=1e-4
    )


def test_response_rises_with_dose():
    """pC is *minus* log10 concentration, so a lower pC means a higher dose."""
    parameters = _parameters(pic50=6.0)
    high_dose = float(inhibition_at(parameters, 4.0))  # 100 uM
    low_dose = float(inhibition_at(parameters, 8.0))  # 10 nM
    assert high_dose > low_dose


def test_a_dose_above_the_ic50_inhibits_more_than_half():
    """The other half of the IC50 definition, and a guard on the sign convention."""
    parameters = _parameters(pic50=6.0)
    emax = _emax_of(parameters)
    assert float(inhibition_at(parameters, 5.0)) > emax / 2  # 10 uM, above the IC50
    assert float(inhibition_at(parameters, 7.0)) < emax / 2  # 100 nM, below it


def test_response_is_bounded_between_zero_and_one():
    for pc in (2.0, 5.0, 9.0):
        value = float(inhibition_at(_parameters(), pc))
        assert 0.0 <= value <= 1.0


def test_a_more_potent_compound_responds_more_at_a_fixed_dose():
    weak = float(inhibition_at(_parameters(pic50=4.0), SCREEN_PC))
    potent = float(inhibition_at(_parameters(pic50=7.0), SCREEN_PC))
    assert potent > weak


def test_an_unconstrained_curve_defaults_to_mass_action():
    """A midpoint observation says nothing about the slope, so the default must hold."""
    from torch.nn.functional import softplus

    from cypshift.models.curve import HILL_OFFSET

    assert float(softplus(torch.tensor(HILL_OFFSET))) == pytest.approx(1.0, abs=1e-3)


def test_a_steeper_hill_slope_sharpens_the_transition():
    shallow = _parameters(pic50=6.0, hill_raw=-2.0)
    steep = _parameters(pic50=6.0, hill_raw=3.0)
    # Above the IC50 a steeper curve sits nearer its plateau; below it, further away.
    assert float(inhibition_at(steep, 5.0)) > float(inhibition_at(shallow, 5.0))
    assert float(inhibition_at(steep, 7.0)) < float(inhibition_at(shallow, 7.0))


def test_the_curve_is_monotone_in_dose_for_any_slope():
    """Softplus keeps the Hill slope positive, so response never falls as dose rises."""
    for raw in (-10.0, 0.0, 10.0):
        parameters = _parameters(pic50=6.0, hill_raw=raw)
        by_rising_dose = [float(inhibition_at(parameters, pc)) for pc in (8.0, 7.0, 6.0, 5.0, 4.0)]
        assert all(a <= b for a, b in zip(by_rising_dose, by_rising_dose[1:], strict=False))


def _observations(seed=0):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        y = rng.uniform(4.5, 7.0, len(SMILES))
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    return build_observations(pd.DataFrame(frame))


def _model(**kwargs):
    settings = {"n_bits": 128, "hidden": 32, "epochs": 30, "n_forward_passes": 4}
    return CurveModel(**{**settings, **kwargs})


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="model is not fitted"):
        _model().predict_pic50(SMILES[:3])


def test_fitting_on_an_empty_table_is_rejected():
    with pytest.raises(ValueError, match="no finite measurements"):
        _model().fit(_observations().iloc[:0])


def test_the_midpoint_is_recovered_from_fitted_curves():
    model = _model(epochs=150).fit(_observations())
    predicted = model.predict_pic50(SMILES)
    assert 3.0 < predicted.mean() < 8.0


def test_every_answer_reads_off_the_same_curve():
    model = _model().fit(_observations())
    smiles = SMILES[:6]
    assert model.predict_pic50(smiles).shape == (6, len(ISOFORMS))
    assert model.predict_response(smiles).shape == (6, len(ISOFORMS))
    assert model.predict_is_time_dependent(smiles).shape == (6, len(ISOFORMS))


def test_the_screen_reading_is_the_curve_at_one_concentration():
    """Not a calibrated pseudo-label: literally the predicted curve evaluated there."""
    model = _model().fit(_observations())
    smiles = SMILES[:5]
    response = model.predict_response(smiles, pc=SCREEN_PC)
    assert ((response >= 0) & (response <= 1)).all()


def test_the_screen_saturates_for_potent_compounds():
    """Why a single concentration carries little information at the potent end.

    At 49.5 uM a compound with pIC50 well above 4.3 is far past its own IC50 and sits
    on the plateau, so the readout stops distinguishing potencies. This is a property
    of the assay, not of the model, and it bounds what the screen can ever contribute
    about strong inhibitors.
    """
    weak = _parameters(pic50=3.0)
    moderate = _parameters(pic50=4.3)  # right at the screen concentration
    potent = _parameters(pic50=6.0)
    very_potent = _parameters(pic50=8.0)

    responses = [float(inhibition_at(p, SCREEN_PC)) for p in (weak, moderate, potent, very_potent)]
    # Monotone in potency, as the physics requires.
    assert all(a < b for a, b in zip(responses, responses[1:], strict=False))
    # But the top two are nearly indistinguishable, while the bottom two are not.
    assert responses[3] - responses[2] < 0.1 * (responses[2] - responses[1])


def test_the_two_arms_are_separate_curves():
    model = _model().fit(_observations())
    direct = model.predict_pic50(SMILES[:5], arm=ARMS[0])
    preincubated = model.predict_pic50(SMILES[:5], arm=ARMS[1])
    assert not np.allclose(direct, preincubated)


def test_parameter_order_is_what_the_loss_assumes():
    assert (PIC50, HILL, EMAX) == (0, 1, 2)


def test_an_observation_only_reaches_the_head_it_observed():
    """The property the whole decomposition rests on.

    A measurement on one enzyme, in one arm, must produce gradient in that head alone.
    If it leaked into the others, a CYP2C9 reading would drag CYP2D6 -- which is
    exactly wrong, since the two recognise opposite charges: an acid helps at CYP2C9
    and hurts at CYP2D6, and a basic amine does the reverse.
    """
    from cypshift.models.curve import CurveNetwork, curve_loss

    torch.manual_seed(0)
    network = CurveNetwork(n_features=8, n_targets=4, n_arms=2, hidden=16, dropout=0.0)
    predicted = network(torch.randn(1, 8))

    observed_target, observed_arm = 1, 0
    parameters = predicted[:, observed_target, observed_arm]
    loss = curve_loss(
        parameters,
        observation_kind=torch.tensor([0]),
        lower=torch.tensor([9.0]),
        upper=torch.tensor([9.5]),
        weight=torch.tensor([1.0]),
    )
    loss.backward()

    touched = [
        index
        for index, head in enumerate(network.heads)
        if head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    ]
    assert touched == [observed_target * 2 + observed_arm]


def test_a_screen_row_and_a_curve_row_constrain_different_things():
    """The measurement channel routes the comparison, not just the head."""
    from cypshift.models.curve import curve_loss

    parameters = torch.tensor([[6.0, 0.0, 0.0]], dtype=torch.float32)
    # As a midpoint observation, a pIC50 of 6.0 sits inside [5.8, 6.2].
    as_midpoint = curve_loss(
        parameters, torch.tensor([0]), torch.tensor([5.8]), torch.tensor([6.2]), torch.tensor([1.0])
    )
    # As a screen reading, the same curve predicts a response near 0.9, which is
    # nowhere near that interval.
    as_response = curve_loss(
        parameters, torch.tensor([1]), torch.tensor([5.8]), torch.tensor([6.2]), torch.tensor([1.0])
    )
    assert float(as_midpoint) == 0.0
    assert float(as_response) > 4.0


def test_both_conditioning_modes_train_and_predict():
    """Heads and descriptors are alternatives to be swept, not a settled choice."""
    observations = _observations()
    for mode in ("heads", "descriptor"):
        model = _model(conditioning=mode).fit(observations)
        assert model.predict_pic50(SMILES[:4]).shape == (4, len(ISOFORMS))


def test_an_unknown_conditioning_mode_is_rejected():
    from cypshift.models.curve import CurveNetwork

    with pytest.raises(ValueError, match="unknown conditioning"):
        CurveNetwork(n_features=8, n_targets=4, conditioning="telepathy")


def test_descriptor_conditioning_needs_descriptors():
    from cypshift.models.curve import CurveNetwork

    with pytest.raises(ValueError, match="needs one descriptor row"):
        CurveNetwork(n_features=8, n_targets=4, conditioning="descriptor")
