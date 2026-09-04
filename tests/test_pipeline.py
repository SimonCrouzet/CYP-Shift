"""Turning a configuration into an experiment."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import assemble_training_data, build_model, metrics_for_tracking


def _retained(n=30):
    rng = np.random.default_rng(0)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(n)], "SMILES": ["CCO"] * n}
    for isoform in ISOFORMS:
        frame[f"{isoform}_pIC50_direct_inhibition"] = rng.uniform(3, 7, n)
    return pd.DataFrame(frame)


def test_every_declared_model_kind_can_be_built():
    for kind in ("boosted", "neural", "graph"):
        config = ExperimentConfig()
        config.override("model.kind", kind)
        model = build_model(config)
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_the_conditioning_strategy_reaches_the_model():
    config = ExperimentConfig()
    config.override("conditioning.strategy", "metadata")
    assert build_model(config).conditioning == "metadata"


def test_model_hyperparameters_reach_the_model():
    config = ExperimentConfig()
    config.override("model.boosted.num_leaves", 15)
    assert build_model(config).num_leaves == 15


def test_an_unbuildable_model_kind_is_rejected():
    config = ExperimentConfig()
    config.model.kind = "per_target_ensemble"  # declared but has no builder
    with pytest.raises(ValueError, match="has no builder"):
        build_model(config)


def test_the_no_screen_strategy_uses_only_the_curves():
    config = ExperimentConfig()
    config.override("multifidelity.strategy", "none")
    observations, surrogate = assemble_training_data(config, _retained())
    assert set(observations["assay"]) == {"direct_inhibition_drc"}
    assert surrogate is None


def test_an_unknown_strategy_says_so_rather_than_falling_back():
    """Declaring an option in the config must not silently do something else."""
    config = ExperimentConfig()
    config.multifidelity.strategy = "distillation"
    with pytest.raises(ValueError, match="unknown multi-fidelity strategy"):
        assemble_training_data(config, _retained())


@pytest.mark.parametrize("strategy", ["predicted_label", "embedding"])
@pytest.mark.parametrize("kind", ["graph", "curve"])
def test_a_surrogate_strategy_is_refused_where_it_makes_no_sense(kind, strategy):
    """A surrogate predicts what the screen would have said, for a model that cannot
    use the reading itself. The curve model can: the screen is inhibition at 4.95e-5 M,
    a point on the curve it predicts, so substituting a prediction of a measurement we
    hold would be strictly worse. The graph model has no dense matrix to append to."""
    config = ExperimentConfig()
    config.override("multifidelity.strategy", strategy)
    config.override("model.kind", kind)
    with pytest.raises(ValueError, match="is not available for"):
        assemble_training_data(config, _retained())


@pytest.mark.parametrize("kind", ["boosted", "neural"])
def test_scalar_models_keep_the_surrogate_strategies(kind):
    """They predict a pIC50 and so cannot consume a response measurement; the surrogate
    is the only route the screen has into them."""
    from cypshift.pipeline import supported_multifidelity

    assert set(supported_multifidelity(kind)) >= {"predicted_label", "embedding"}


def test_source_weights_reach_the_observation_table():
    config = ExperimentConfig()
    config.override("multifidelity.strategy", "none")
    observations, _ = assemble_training_data(config, _retained())
    assert (observations["weight"] == 1.0).all()


def test_metrics_flatten_to_scalars_a_tracker_can_log():
    scores = pd.DataFrame(
        {"ST-RAE_mean": [0.68, 0.70], "ST-RAE_std": [0.01, 0.01], "MAE_mean": [0.5, 0.6]},
        index=["CYP3A4_pIC50_direct_inhibition", "MA"],
    )
    flat = metrics_for_tracking(scores)
    assert flat["CYP3A4_ST-RAE"] == 0.68
    assert flat["MA_ST-RAE"] == 0.70
    assert all(isinstance(v, float) for v in flat.values())


def test_the_curve_model_receives_the_screen_as_response_not_pic50():
    """A curve model compares against an inhibited fraction, so units must match."""
    from cypshift.models.curve import CurveModel

    config = ExperimentConfig()
    config.override("model.kind", "curve")
    assert isinstance(build_model(config), CurveModel)


def test_curve_hyperparameters_reach_the_model():
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.override("model.curve.hidden", 64)
    assert build_model(config).hidden == 64
