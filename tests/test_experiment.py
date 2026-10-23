"""Experiment configuration and the conditioning strategies it selects."""

import json

import pandas as pd
import pytest

from cypshift.data.assembly import N_METADATA_COLUMNS, encode_conditions, n_condition_columns
from cypshift.experiment import (
    CONDITIONING_STRATEGIES,
    CURVE_CONDITIONING,
    SCREEN_STRATEGIES,
    ExperimentConfig,
    Stage,
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
    # metadata is the target one-hot followed by the assay property scalars, of which
    # preincubated is the first -- not an assay one-hot.
    encoded = encode_conditions(OBSERVATIONS, "metadata")
    preincubated = encoded[:, -N_METADATA_COLUMNS]
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
    assert set(SCREEN_STRATEGIES) == {
        "extra_rows",
        "predicted_label",
        "embedding",
        "response_head",
        "log2fc_response",
    }


class TestCapabilityContract:
    """Settings a run would accept then silently ignore must be rejected up front (F8)."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"auxiliary.scored_weight": -1.0},
            {"sources.structure": True, "model.kind": "neural"},
            {"model.kind": "graph", "features.molecular_embedding": "mammal"},
            {"model.kind": "graph", "training.preset": "coarse_to_fine"},
            {"model.kind": "curve", "model.objective": "absolute"},
            # The analysis-note avenues are graph-only, or mutually exclusive, so a run that
            # could not apply the setting is refused rather than credited with it.
            {"model.kind": "neural", "model.point_loss_weight": 0.5},
            {"model.kind": "boosted", "model.ranking_loss_weight": 0.5},
            {"model.kind": "boosted", "model.graph.adapt_target": "CYP2D6"},
            {"model.kind": "boosted", "multifidelity.screen_strategy": "response_head"},
            {"model.kind": "boosted", "model.graph.per_target_heads": True},
            {
                "model.kind": "graph",
                "model.graph.separate_trunks": True,
                "model.graph.adapt_target": "CYP2D6",
            },
            {
                "model.kind": "graph",
                "model.graph.separate_trunks": True,
                "model.graph.per_target_heads": True,
            },
            {
                "model.kind": "graph",
                "model.graph.adapt_target": "CYP2D6",
                "model.graph.per_target_heads": True,
            },
            # The (target, measurement) table is an alternative readout on the shared trunk, and
            # its gate reads the factorised one-hot layout.
            {"model.kind": "boosted", "model.graph.per_measurement_heads": True},
            {
                "model.kind": "graph",
                "model.graph.per_measurement_heads": True,
                "model.graph.per_target_heads": True,
            },
            {
                "model.kind": "graph",
                "model.graph.per_measurement_heads": True,
                "model.graph.adapt_target": "CYP2D6",
            },
            {
                "model.kind": "graph",
                "model.graph.per_measurement_heads": True,
                "model.graph.separate_trunks": True,
            },
            {
                "model.kind": "graph",
                "model.graph.per_measurement_heads": True,
                "conditioning.strategy": "metadata",
            },
            # The point/ranking terms layer onto the soft-threshold loss only.
            {
                "model.kind": "graph",
                "model.objective": "squared",
                "model.point_loss_weight": 0.5,
            },
        ],
    )
    def test_silently_ignored_combinations_are_rejected(self, overrides):
        config = ExperimentConfig.load()
        for key, value in overrides.items():
            config.override(key, value)
        with pytest.raises(ValueError):
            config.validate()

    def test_supported_combinations_still_validate(self):
        for overrides in (
            {"model.kind": "graph"},
            {"sources.structure": True},
            {"model.kind": "graph", "multifidelity.screen_strategy": "response_head"},
            # The screen fraction and the qHTS max-response now coexist (one response column each).
            {
                "model.kind": "graph",
                "sources.veith": True,
                "sources.qhts_representation": "max_response",
                "multifidelity.screen_strategy": "response_head",
            },
            {"model.kind": "graph", "model.graph.adapt_target": "CYP2D6"},
            {"model.kind": "graph", "model.graph.per_target_heads": True},
            {"model.kind": "graph", "model.graph.per_measurement_heads": True},
            # The full panel the table is for: screen + both Veith quantities on the table.
            {
                "model.kind": "graph",
                "model.graph.per_measurement_heads": True,
                "sources.veith": True,
                "sources.qhts_representation": "both",
                "multifidelity.screen_strategy": "response_head",
            },
            {
                "model.kind": "graph",
                "model.point_loss_weight": 0.5,
                "model.ranking_loss_weight": 0.5,
            },
        ):
            config = ExperimentConfig.load()
            for key, value in overrides.items():
                config.override(key, value)
            config.validate()  # no raise

    @pytest.mark.parametrize(
        "stage",
        [
            Stage(freeze_trunk=True),
            Stage(epochs=5),
            Stage(lr=1e-4),
            Stage(assays=("direct_inhibition_drc",)),
            Stage(data_spaces=("challenge",)),
        ],
    )
    def test_graph_rejects_a_non_default_single_training_stage(self, stage):
        # The graph factory never applies per-stage settings, so a single non-default stage
        # would silently do nothing; validation rejects it rather than crediting a run with a
        # curriculum it never ran.
        config = ExperimentConfig.load()
        config.override("model.kind", "graph")
        config.training.stages = [stage]
        with pytest.raises(ValueError, match="per-stage training settings"):
            config.validate()

    def test_graph_accepts_the_default_single_stage(self):
        config = ExperimentConfig.load()
        config.override("model.kind", "graph")
        config.training.stages = [Stage()]
        config.validate()  # the trivial pooled stage is what the graph actually runs
