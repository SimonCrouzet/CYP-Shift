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
        ("model.graph.matmul_precision", "approximate"),
        ("auxiliary.row_selection", "telepathy"),
        ("model.graph.aggregation", "median"),
    ],
)
def test_an_unknown_option_is_rejected(field, value):
    config = ExperimentConfig()
    config.override(field, value)
    with pytest.raises(ValueError, match="is not one of"):
        config.validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model.graph.early_stopping_patience": -0.1}, "share of the epochs"),
        ({"model.graph.early_stopping_patience": 3}, "share of the epochs"),
        ({"model.graph.prune_tolerance": -0.1}, "non-negative"),
        ({"model.graph.prune_reference": "bundle"}, "monitor_fold"),
        ({"model.graph.monitor_fold": True, "model.kind": "boosted"}, "model.kind='graph'"),
        (
            {"model.graph.monitor_fold": True, "model.graph.separate_trunks": True},
            "model.kind='graph'",
        ),
    ],
)
def test_impossible_fold_monitoring_settings_are_rejected(overrides, match):
    config = ExperimentConfig.load()
    config.override("model.kind", "graph")
    for key, value in overrides.items():
        config.override(key, value)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_the_monitoring_defaults_are_a_tenth_of_the_epochs_and_a_30_percent_tolerance():
    graph = ExperimentConfig().model.graph
    assert (graph.monitor_fold, graph.early_stopping_patience, graph.prune_tolerance) == (
        False,
        0.1,
        0.3,
    )


def test_fold_monitoring_with_early_stopping_and_pruning_is_accepted():
    config = ExperimentConfig.load()
    config.override("model.kind", "graph")
    config.override("model.graph.monitor_fold", True)
    config.override("model.graph.early_stopping_patience", 0.2)
    config.override("model.graph.prune_reference", "data/run_bundles/x")
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
            {
                "model.kind": "graph",
                "model.graph.pretraining": "mammal",
                "features.molecular_embedding": "chemeleon",
            },
            {"model.kind": "curve", "model.objective": "absolute"},
            # The analysis-note avenues are graph-only, or mutually exclusive, so a run that
            # could not apply the setting is refused rather than credited with it.
            {"model.kind": "neural", "model.point_loss_weight": 0.5},
            {"model.kind": "boosted", "model.ranking_loss_weight": 0.5},
            {"model.kind": "boosted", "model.graph.adapt_target": "CYP2D6"},
            {"model.kind": "boosted", "multifidelity.screen_strategy": "response_head"},
            {"model.kind": "boosted", "model.graph.per_target_heads": True},
            {"model.kind": "boosted", "model.graph.finetune_per_target_epochs": 5},
            {"model.kind": "boosted", "model.graph.save_embedding": True},
            {"model.kind": "graph", "model.graph.tdi_label_loss": "bce"},  # no tdi_label source
            {"model.kind": "graph", "model.graph.tdi_label_loss": "hinge"},
            {
                "model.kind": "graph",
                "sources.tdi_label": True,
                "model.graph.tdi_label_loss": "bce",
                "model.graph.tdi_label_pos_weight": 0.0,
            },
            {
                "model.kind": "graph",
                "model.graph.save_embedding": True,
                "model.graph.separate_trunks": True,
            },
            {"model.kind": "graph", "model.graph.finetune_per_target_epochs": -1},
            {
                "model.kind": "graph",
                "model.graph.finetune_per_target_epochs": 5,
                "model.graph.separate_trunks": True,
            },
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

    @pytest.mark.parametrize("kind", ["boosted", "neural", "curve"])
    @pytest.mark.parametrize("representation", ["max_response", "both"])
    def test_max_response_is_a_data_side_choice_for_models_without_a_response_head(
        self, kind, representation
    ):
        config = ExperimentConfig.load()
        for key, value in {
            "model.kind": kind,
            "sources.veith": True,
            "sources.qhts_representation": representation,
        }.items():
            config.override(key, value)
        config.validate()  # no raise

    @pytest.mark.parametrize("kind", ["graph", "boosted", "neural", "curve"])
    @pytest.mark.parametrize("source", ["veith", "tox21"])
    def test_max_response_takes_any_qhts_source(self, kind, source):
        config = ExperimentConfig.load()
        for key, value in {
            "model.kind": kind,
            f"sources.{source}": True,
            "sources.qhts_representation": "max_response",
        }.items():
            config.override(key, value)
        config.validate()  # no raise

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.kind": "boosted", "sources.qhts_representation": "max_response"},
            {
                "model.kind": "curve",
                "model.curve.backbone": "graph",
                "sources.veith": True,
                "sources.qhts_representation": "max_response",
            },
        ],
    )
    def test_max_response_still_needs_veith_and_a_model_that_can_read_it(self, overrides):
        config = ExperimentConfig.load()
        for key, value in overrides.items():
            config.override(key, value)
        with pytest.raises(ValueError, match="qhts_representation"):
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
        "preset", ["direct_finetune", "direct_finetune_frozen", "pooled_continue"]
    )
    def test_graph_accepts_a_multi_stage_curriculum(self, preset):
        config = ExperimentConfig.load()
        config.override("model.kind", "graph")
        config.training.preset = preset
        config.validate()
        assert len(config.training.effective_stages()) == 2

    def test_graph_accepts_the_default_single_stage(self):
        config = ExperimentConfig.load()
        config.override("model.kind", "graph")
        config.training.stages = [Stage()]
        config.validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"auxiliary.row_selection": "clusters"},
        {"auxiliary.max_rows_per_group": 2000},
        {"auxiliary.row_selection": "maxmin", "auxiliary.max_rows_per_group": -1},
    ],
)
def test_a_row_selection_and_its_cap_go_together(overrides):
    config = ExperimentConfig.load()
    for key, value in overrides.items():
        config.override(key, value)
    with pytest.raises(ValueError, match="go together"):
        config.validate()


def test_a_row_selection_with_a_cap_is_accepted():
    config = ExperimentConfig.load()
    config.override("auxiliary.row_selection", "nearest")
    config.override("auxiliary.max_rows_per_group", 2000)
    config.validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"auxiliary.row_selection": "random", "auxiliary.row_share": 1.5},
        {
            "auxiliary.row_selection": "random",
            "auxiliary.row_share": 0.5,
            "auxiliary.max_rows_per_group": 100,
        },
        {"auxiliary.row_share": 0.5},
    ],
)
def test_a_row_selection_takes_exactly_one_size(overrides):
    config = ExperimentConfig.load()
    for key, value in overrides.items():
        config.override(key, value)
    with pytest.raises(ValueError, match="go together"):
        config.validate()


def test_a_row_share_with_a_listed_assay_is_accepted_and_an_unknown_assay_is_not():
    config = ExperimentConfig.load()
    config.override("auxiliary.row_selection", "random")
    config.override("auxiliary.row_share", 0.5)
    config.override(
        "auxiliary.row_selection_assays", ["fluorogenic_pic50", "luminescent_max_response"]
    )
    config.validate()
    config.override("auxiliary.row_selection_assays", ["telepathy"])
    with pytest.raises(ValueError, match="unknown"):
        config.validate()


def test_the_readout_must_have_at_least_one_hidden_layer():
    config = ExperimentConfig()
    config.override("model.graph.ffn_layers", 0)
    with pytest.raises(ValueError, match="ffn_layers"):
        config.validate()


def test_the_graph_accepts_a_frozen_embedding_as_a_feature_block():
    config = ExperimentConfig()
    for key, value in {"model.kind": "graph", "features.molecular_embedding": "chemeleon"}.items():
        config.override(key, value)
    config.validate()


class TestSharedHiddenReadoutOptions:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.kind": "boosted", "model.graph.hard_parameter_sharing": True},
            {"model.kind": "boosted", "model.graph.per_task_scaling": True},
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "model.graph.per_measurement_heads": True,
            },
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "model.graph.per_target_heads": True,
            },
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "model.graph.adapt_target": "CYP2D6",
            },
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "model.graph.separate_trunks": True,
            },
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "conditioning.strategy": "metadata",
            },
        ],
    )
    def test_combinations_that_would_be_ignored_are_rejected(self, overrides):
        config = ExperimentConfig.load()
        for key, value in overrides.items():
            config.override(key, value)
        with pytest.raises(ValueError):
            config.validate()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model.kind": "graph", "model.graph.hard_parameter_sharing": True},
            {"model.kind": "graph", "model.graph.per_task_scaling": True},
            {
                "model.kind": "graph",
                "model.graph.hard_parameter_sharing": True,
                "model.graph.per_task_scaling": True,
                "model.graph.finetune_per_target_epochs": 10,
            },
            {
                "model.kind": "graph",
                "model.graph.per_task_scaling": True,
                "model.graph.per_measurement_heads": True,
            },
        ],
    )
    def test_the_options_validate_on_the_graph(self, overrides):
        config = ExperimentConfig.load()
        for key, value in overrides.items():
            config.override(key, value)
        config.validate()

    def test_a_config_saved_before_the_options_existed_loads_with_them_off(self):
        saved = ExperimentConfig.load().to_dict()
        del saved["model"]["graph"]["hard_parameter_sharing"]
        del saved["model"]["graph"]["per_task_scaling"]
        loaded = ExperimentConfig.from_dict(saved)
        assert not loaded.model.graph.hard_parameter_sharing
        assert not loaded.model.graph.per_task_scaling


class TestPerTargetFineTuneAssays:
    def _config(self, **graph):
        config = ExperimentConfig.load()
        config.override("model.kind", "graph")
        for key, value in graph.items():
            config.override(f"model.graph.{key}", value)
        return config

    def test_the_assays_need_a_fine_tune_and_known_names(self):
        with pytest.raises(ValueError, match="ignored without"):
            self._config(finetune_per_target_assays=("direct_inhibition_drc",)).validate()
        with pytest.raises(ValueError, match="unknown"):
            self._config(
                finetune_per_target_epochs=10, finetune_per_target_assays=("not_an_assay",)
            ).validate()
        self._config(
            finetune_per_target_epochs=10, finetune_per_target_assays=("direct_inhibition_drc",)
        ).validate()

    def test_a_config_saved_before_the_field_existed_loads_with_all_assays(self):
        saved = ExperimentConfig.load().to_dict()
        del saved["model"]["graph"]["finetune_per_target_assays"]
        assert not ExperimentConfig.from_dict(saved).model.graph.finetune_per_target_assays
