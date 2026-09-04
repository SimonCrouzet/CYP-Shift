"""Experiment configuration and the conditioning strategies it selects."""

import json

import pandas as pd
import pytest

from cypshift.data.assembly import encode_conditions, n_condition_columns
from cypshift.experiment import (
    CONDITIONING_STRATEGIES,
    CURVE_CONDITIONING,
    MULTIFIDELITY_STRATEGIES,
    ExperimentConfig,
)

OBSERVATIONS = pd.DataFrame(
    {
        "target": ["CYP1A2", "CYP3A4", "CYP2D6"],
        "assay": ["direct_inhibition_drc", "tdi_drc", "direct_inhibition_screen"],
    }
)


def test_defaults_validate():
    assert ExperimentConfig().validate() is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("conditioning.strategy", "telepathy"),
        ("model.curve.conditioning", "telepathy"),
        ("multifidelity.strategy", "wishing"),
        ("model.kind", "oracle"),
        ("model.objective", "vibes"),
    ],
)
def test_an_unknown_option_is_rejected(field, value):
    config = ExperimentConfig()
    config.override(field, value)
    with pytest.raises(ValueError, match="is not one of"):
        config.validate()


def test_an_impossible_holdout_fraction_is_rejected():
    config = ExperimentConfig()
    config.override("data.holdout_fraction", 1.5)
    with pytest.raises(ValueError, match="strictly between"):
        config.validate()


def test_overriding_an_unknown_field_is_an_error():
    with pytest.raises(KeyError, match="no such configuration"):
        ExperimentConfig().override("model.not_a_field", 3)


def test_nested_override_reaches_a_model_specific_section():
    config = ExperimentConfig()
    config.override("model.boosted.num_leaves", 15)
    assert config.model.boosted.num_leaves == 15


def test_round_trips_through_json(tmp_path):
    config = ExperimentConfig()
    config.override("model.kind", "neural")
    path = config.save(tmp_path / "config.json")
    assert ExperimentConfig.from_dict(json.loads(path.read_text())).model.kind == "neural"


def test_flat_keys_are_dotted_for_sweep_tools():
    flat = ExperimentConfig().to_flat_dict()
    assert "model.boosted.num_leaves" in flat
    assert "conditioning.strategy" in flat
    assert all("." in key or key.count(".") == 0 for key in flat)


def test_load_applies_overrides_with_double_underscores(tmp_path):
    path = ExperimentConfig().save(tmp_path / "c.json")
    config = ExperimentConfig.load(path, model__kind="graph")
    assert config.model.kind == "graph"


def test_every_declared_conditioning_strategy_encodes():
    for strategy in CONDITIONING_STRATEGIES:
        encoded = encode_conditions(OBSERVATIONS, strategy)
        assert encoded.shape == (len(OBSERVATIONS), n_condition_columns(strategy))


def test_factorised_conditioning_shares_a_condition_across_targets():
    """The point of factorising: 'preincubated' is one direction common to every enzyme."""
    rows = pd.DataFrame({"target": ["CYP1A2", "CYP3A4"], "assay": ["tdi_drc", "tdi_drc"]})
    encoded = encode_conditions(rows, "factorised")
    assay_block = encoded[:, 4:]
    assert (assay_block[0] == assay_block[1]).all()


def test_metadata_conditioning_describes_the_experiment():
    encoded = encode_conditions(OBSERVATIONS, "metadata")
    preincubated = encoded[:, -3]
    assert preincubated[0] == 0.0 and preincubated[1] == 1.0


def test_an_unknown_conditioning_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown conditioning strategy"):
        encode_conditions(OBSERVATIONS, "osmosis")


def test_input_encodings_and_structural_conditioning_are_separate_axes():
    """One is columns appended to the input, the other is how the model is wired."""
    assert set(CONDITIONING_STRATEGIES).isdisjoint(CURVE_CONDITIONING)
    assert set(CURVE_CONDITIONING) == {"heads", "descriptor", "film"}


def test_all_multifidelity_strategies_are_named():
    assert set(MULTIFIDELITY_STRATEGIES) == {"none", "extra_rows", "predicted_label", "embedding"}
