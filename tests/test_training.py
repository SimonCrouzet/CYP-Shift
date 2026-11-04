"""The staged training curriculum shared across the model families.

A stage selects data on two facets -- proximity ``data_spaces`` and data-type ``assays`` -- and
carries a swappable strategy (``epochs``, ``lr``, ``freeze_trunk``). The stage sequence is
shared; each model applies its own transition mechanism.
"""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS
from cypshift.data.assembly import DATA_SPACES, build_observations
from cypshift.experiment import CURRICULUM_PRESETS, ExperimentConfig, Stage, TrainingConfig
from cypshift.models.base import ResolvedStage, active_stages, rows_for_stage, training_stages
from cypshift.models.boosted import BoostedTrees
from cypshift.models.neural import NeuralNetwork

SCAFFOLDS = ["CCOc1ccc({})cc1", "c1ccc2[nH]c({})cc2c1", "Cn1cnc({})c1"]
SMILES = [s.format(x) for s in SCAFFOLDS for x in ["C", "CC", "Cl", "Br", "F", "O", "N", "CCC"]]


def _challenge(seed=0):
    rng = np.random.default_rng(seed)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(len(SMILES))], "SMILES": SMILES}
    for isoform in ISOFORMS:
        y = rng.uniform(3.0, 7.0, len(SMILES))
        frame[f"{isoform}_pIC50_direct_inhibition"] = y
        frame[f"{isoform}_pIC50_direct_inhibition_conf_low"] = y - 0.2
        frame[f"{isoform}_pIC50_direct_inhibition_conf_high"] = y + 0.2
    return build_observations(pd.DataFrame(frame))


def _observations(seed=0):
    """Challenge dose-response rows plus a block of distant-auxiliary rows."""
    challenge = _challenge(seed)
    aux = challenge.head(12).copy()
    aux["data_space"] = "distant_aux"
    aux["weight"] = 0.4
    return pd.concat([challenge, aux], ignore_index=True)


def _two_stage():
    return TrainingConfig(
        stages=[
            Stage(data_spaces=("distant_aux", "near_aux"), epochs=2),
            Stage(
                data_spaces=("challenge",),
                assays=("direct_inhibition_drc",),
                epochs=2,
                freeze_trunk=True,
            ),
        ]
    )


def test_training_stages_default_is_one_pooled_stage_over_every_data_space():
    assert training_stages(None, 60, 1e-3) == [ResolvedStage(DATA_SPACES, None, 60, 1e-3, False)]


def test_a_stage_reuses_the_models_epochs_and_lr_when_unset():
    config = TrainingConfig(stages=[Stage(data_spaces=("challenge",))])
    resolved = training_stages(config, 40, 0.05)
    assert resolved[0].epochs == 40 and resolved[0].lr == pytest.approx(0.05)


def test_rows_for_stage_filters_on_both_facets():
    observations = _observations()
    challenge_drc = ResolvedStage(("challenge",), ("direct_inhibition_drc",), 1, 1e-3, False)
    mask = rows_for_stage(observations, challenge_drc)
    assert (observations.loc[mask, "data_space"] == "challenge").all()
    # the aux rows share the assay but not the data_space, so the facet pair excludes them
    aux_drc = ResolvedStage(("distant_aux",), ("direct_inhibition_drc",), 1, 1e-3, False)
    assert rows_for_stage(observations, aux_drc).sum() == 12


def test_active_stages_drop_facets_with_no_rows_this_run():
    observations = _challenge()  # challenge data_space only, direct_inhibition_drc only
    stages = training_stages(_two_stage(), 5, 1e-3)
    kept = active_stages(observations, stages)
    assert [s.data_spaces for s in kept] == [("challenge",)]


def test_active_stages_drop_an_absent_assay_facet():
    observations = _challenge()  # no TDI rows
    tdi_only = training_stages(
        TrainingConfig(stages=[Stage(data_spaces=("challenge",), assays=("tdi_drc",))]), 5, 1e-3
    )
    # every stage empty -> fall back to one pooled stage over all data_spaces
    assert active_stages(observations, tdi_only)[0].assays is None


def test_preset_selects_a_named_curriculum():
    training = TrainingConfig(preset="coarse_to_fine")
    assert training.effective_stages() == CURRICULUM_PRESETS["coarse_to_fine"]
    assert len(training.effective_stages()) == 3


def test_preset_overrides_an_explicit_stage_list():
    training = TrainingConfig(preset="pooled", stages=[Stage(data_spaces=("challenge",))])
    assert training.effective_stages() == CURRICULUM_PRESETS["pooled"]


def test_every_preset_validates():
    for name in CURRICULUM_PRESETS:
        ExperimentConfig(training=TrainingConfig(preset=name)).validate()


def test_unknown_preset_is_rejected_by_validation():
    with pytest.raises(ValueError, match="preset"):
        ExperimentConfig(training=TrainingConfig(preset="nope")).validate()


def test_unknown_data_space_is_rejected_by_validation():
    config = ExperimentConfig()
    config.training = TrainingConfig(stages=[Stage(data_spaces=("no_such_data_space",))])
    with pytest.raises(ValueError, match="unknown"):
        config.validate()


def test_unknown_assay_is_rejected_by_validation():
    config = ExperimentConfig()
    config.training = TrainingConfig(stages=[Stage(assays=("no_such_assay",))])
    with pytest.raises(ValueError, match="unknown"):
        config.validate()


def test_boosted_runs_a_two_stage_curriculum_and_predicts():
    # A built-in objective on the tiny per-stage data: the soft-threshold custom
    # objective needs more rows than a toy stage carries (a LightGBM constraint, not ours).
    model = BoostedTrees(
        n_bits=128,
        n_estimators=8,
        n_members=2,
        num_leaves=7,
        loss="absolute",
        training=_two_stage(),
    )
    model.fit(_observations())
    prediction = model.predict(SMILES[:4])
    assert prediction.mean.shape == (4, len(model.endpoints))
    assert np.isfinite(prediction.mean).all()


def test_calibration_recovers_a_known_per_assay_offset():
    """A shared potency plus a per-assay bias: the reference is pinned to zero and an
    auxiliary assay measured a fixed offset above it learns that offset, so predictions
    come back on the reference scale rather than the pooled average."""
    rng = np.random.default_rng(0)
    rows = []
    for i, smiles in enumerate(SMILES):
        y = rng.uniform(4.0, 7.0)
        rows.append(
            dict(
                Molecule_Name=f"m{i}",
                SMILES=smiles,
                target="CYP3A4",
                assay="direct_inhibition_drc",
                value=y,
                conf_low=y,
                conf_high=y,
                weight=1.0,
                is_censored=False,
                data_space="challenge",
            )
        )
        rows.append(
            dict(
                Molecule_Name=f"a{i}",
                SMILES=smiles,
                target="CYP3A4",
                assay="luminescent_ac50",
                value=y + 1.5,
                conf_low=y + 1.5,
                conf_high=y + 1.5,
                weight=1.0,
                is_censored=False,
                data_space="near_aux",
            )
        )
    model = NeuralNetwork(
        n_bits=256,
        hidden=32,
        epochs=400,
        learning_rate=3e-3,
        conditioning="calibration",
        seed=0,
        loss="absolute",
    )
    model.fit(pd.DataFrame(rows))
    bias = (
        (model._model.bias.weight - model._model.bias.weight[model._model.reference_index])
        .detach()
        .squeeze(-1)
    )
    assert float(bias[model._model.reference_index]) == pytest.approx(0.0)
    assert float(bias[model.assays.index("luminescent_ac50")]) > 0.8  # heading to 1.5
    assert model.predict(SMILES[:4]).mean.shape == (4, len(model.endpoints))


def test_curriculum_changes_the_neural_fit_and_restores_gradients():
    """The staged path is wired (a different result from pooled) and leaves the trunk
    trainable again after a frozen stage. Uses a head-bearing conditioning so the frozen
    stage has something to train."""
    pooled = NeuralNetwork(n_bits=128, hidden=16, epochs=3, seed=0, conditioning="heads")
    pooled.fit(_observations())
    staged = NeuralNetwork(
        n_bits=128, hidden=16, epochs=3, seed=0, conditioning="heads", training=_two_stage()
    )
    staged.fit(_observations())

    a = pooled.predict(SMILES[:6]).mean
    b = staged.predict(SMILES[:6]).mean
    assert not np.allclose(a, b)
    assert all(parameter.requires_grad for parameter in staged._model.parameters())
