"""Experiment configuration and the conditioning strategies it selects."""

import json

import pandas as pd
import pytest

from cypshift.data.assembly import encode_conditions, n_condition_columns
from cypshift.experiment import (
    CONDITIONING_STRATEGIES,
    CURVE_CONDITIONING,
    SCREEN_STRATEGIES,
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
        ("multifidelity.screen_strategy", "wishing"),
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


def test_every_column_conditioning_strategy_encodes():
    # Only factorised and metadata are *column* encodings that encode_conditions builds.
    # embedding/descriptor/heads/film are model-internal (a learned table, an appended
    # descriptor, per-condition heads, trunk modulation), handled inside the model.
    for strategy in ("factorised", "metadata"):
        encoded = encode_conditions(OBSERVATIONS, strategy)
        assert encoded.shape == (len(OBSERVATIONS), n_condition_columns(strategy))


def test_model_internal_strategies_are_not_column_encodings():
    for strategy in ("embedding", "descriptor", "heads", "film"):
        with pytest.raises(ValueError, match="unknown conditioning strategy"):
            encode_conditions(OBSERVATIONS, strategy)


def test_factorised_conditioning_shares_a_condition_across_targets():
    """The point of factorising: 'preincubated' is one direction common to every enzyme."""
    rows = pd.DataFrame({"target": ["CYP1A2", "CYP3A4"], "assay": ["tdi_drc", "tdi_drc"]})
    encoded = encode_conditions(rows, "factorised")
    assay_block = encoded[:, 4:]
    assert (assay_block[0] == assay_block[1]).all()


def test_metadata_conditioning_describes_the_experiment():
    # metadata is the target one-hot followed by two assay scalars: preincubated, then
    # dose_response. The last two columns, not an assay one-hot.
    encoded = encode_conditions(OBSERVATIONS, "metadata")
    preincubated = encoded[:, -2]
    assert preincubated[0] == 0.0 and preincubated[1] == 1.0


def test_metadata_is_narrower_than_factorised_not_a_superset():
    # metadata replaces the assay one-hot with two scalars, so it must be distinct from
    # and narrower than factorised -- otherwise it would add nothing over it.
    factorised = encode_conditions(OBSERVATIONS, "factorised")
    metadata = encode_conditions(OBSERVATIONS, "metadata")
    assert metadata.shape[1] < factorised.shape[1]


def test_an_unknown_conditioning_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown conditioning strategy"):
        encode_conditions(OBSERVATIONS, "osmosis")


def test_the_curve_conditioning_names_are_shared_with_the_neural_vocabulary():
    """heads/descriptor/film were unified: the neural model offers the same conditioning
    the curve model does, so the names overlap by design. Which model may use which is
    gated per family by supported_conditioning, not by keeping the constants disjoint."""
    assert set(CURVE_CONDITIONING) == {"heads", "descriptor", "film"}
    assert set(CURVE_CONDITIONING) <= set(CONDITIONING_STRATEGIES)


def test_only_the_neural_model_gets_the_full_conditioning_vocabulary():
    from cypshift.pipeline import supported_conditioning

    assert set(supported_conditioning("neural")) == set(CONDITIONING_STRATEGIES)
    assert set(supported_conditioning("boosted")) == {"factorised", "metadata"}
    assert set(supported_conditioning("graph")) == {"factorised", "metadata"}


def test_all_multifidelity_strategies_are_named():
    assert set(SCREEN_STRATEGIES) == {"extra_rows", "predicted_label", "embedding"}
