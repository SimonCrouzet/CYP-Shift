"""The declared sweep: every entry resolves and validates, the 2026-10-24 review's comparison grid
is a clean factorial with matched controls, the priority entries exist, and the run order puts
them (and the regression candidate) first."""

import dataclasses
import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import night_run  # noqa: E402 -- after the scripts path insert, as the other script tests do

from cypshift.experiment import ExperimentConfig  # noqa: E402

#: Pairs of differently named entries that resolve to the same settings, known and kept.
KNOWN_DUPLICATES = {
    frozenset({"tfm_fs_fp_dense", "tfm"}),
    frozenset({"boosted_fs_fp_dense", "boosted"}),
}


def _resolved(overrides: dict) -> ExperimentConfig:
    config = ExperimentConfig()
    for key, value in overrides.items():
        config.override(key, value)
    return config


@pytest.mark.parametrize("phase", ["REGRESSION", "TDI"])
def test_every_declared_config_resolves_and_validates(phase):
    for name, overrides in getattr(night_run, phase).items():
        try:
            _resolved(overrides).validate()
        except Exception as error:  # noqa: BLE001 -- name the entry, whatever raised
            raise AssertionError(f"{phase}[{name!r}] does not validate: {error}") from error


def test_no_two_entries_resolve_to_the_same_settings_beyond_the_known_pairs():
    seen: dict[str, str] = {}
    duplicates = set()
    for name, overrides in night_run.REGRESSION.items():
        key = repr(sorted((k, repr(v)) for k, v in overrides.items()))
        if key in seen:
            duplicates.add(frozenset({name, seen[key]}))
        seen[key] = name
    assert duplicates <= KNOWN_DUPLICATES, duplicates - KNOWN_DUPLICATES


@pytest.mark.parametrize(
    ("pool", "bases"),
    [
        (
            "REGRESSION",
            (
                "neural",
                "neural_octant_tox21",
                "boosted",
                "boosted_octant_tox21",
                "graph",
                "graph_octant_tox21",
                "graph_chemeleon",
            ),
        ),
        ("TDI", ("curve_heads",)),
    ],
)
def test_the_qhts_representation_is_crossed_with_every_family_that_can_read_it(pool, bases):
    declared = getattr(night_run, pool)
    queued = set(getattr(night_run, "QHTS_REGRESSION" if pool == "REGRESSION" else "QHTS_TDI"))
    assert queued <= set(declared)
    for base in bases:
        cells = set()
        for name in queued:
            config = _resolved(declared[name])
            reference = _resolved(declared[base])
            same_base = (
                config.model.kind == reference.model.kind
                and config.sources.octant == reference.sources.octant
                and config.model.graph.pretraining == reference.model.graph.pretraining
            )
            if same_base and config.sources.veith:
                cells.add(
                    (
                        config.sources.qhts_representation,
                        config.auxiliary.include_qhts_inactives,
                    )
                )
        assert cells == {
            ("pic50", True),
            ("pic50", False),
            ("max_response", True),
            ("both", True),
            ("both", False),
        }, (base, cells)


#: Data-side fields no entry varies, each with the reason.
DATA_SIDE_NOT_SWEPT = {
    "data.directory": "location of the challenge files",
    "data.holdout_fraction": "holdout path only; the sweep runs scaffold cross-validation",
    "data.n_similarity_bins": "holdout path only; the sweep runs scaffold cross-validation",
    "data.n_matching_rounds": "holdout path only; the sweep runs scaffold cross-validation",
    "data.seed": "fixed so that configurations are compared on the same folds and seed",
    "sources.structure": "needs the Boltz cofold cache, which is empty",
    "multifidelity.screen_model_hidden": "screen helper width; the graph cannot take that helper",
    "training.preset": "curriculum; the graph ignores stage settings",
    "training.stages": "curriculum; the graph ignores stage settings",
}


def test_every_data_side_field_is_varied_by_some_entry_or_excluded_for_a_stated_reason():
    swept = {key for overrides in night_run.REGRESSION.values() for key in overrides}
    config = ExperimentConfig()
    for section in ("data", "sources", "auxiliary", "multifidelity", "training"):
        for field in dataclasses.fields(getattr(config, section)):
            key = f"{section}.{field.name}"
            assert key in swept or key in DATA_SIDE_NOT_SWEPT, (
                f"{key} is neither swept nor excluded"
            )
    assert not set(DATA_SIDE_NOT_SWEPT) & swept  # an exclusion stays true


def test_the_data_side_group_holds_the_model_fixed_and_covers_the_source_sets():
    names = night_run.DATA_SIDE_REGRESSION
    assert len(set(names)) == len(names) and set(names) <= set(night_run.REGRESSION)
    assert all(_resolved(night_run.REGRESSION[n]).model.kind == "graph" for n in names)
    source_sets = {
        frozenset(
            source
            for source in night_run.AUXILIARY_SOURCES
            if getattr(_resolved(night_run.REGRESSION[n]).sources, source)
        )
        for n in names
    }
    sources = night_run.AUXILIARY_SOURCES
    wanted = {frozenset()} | {
        frozenset(c) for k in (1, 2) for c in itertools.combinations(sources, k)
    }
    wanted |= {frozenset(("octant", "tox21", extra)) for extra in sources[2:]} | {
        frozenset(sources)
    }
    assert wanted <= source_sets
    # cheap configurations first: the number of large sources never decreases along the queue
    heavy = [
        sum(bool(night_run.REGRESSION[n].get(f"sources.{s}")) for s in night_run.HEAVY_SOURCES)
        for n in names
    ]
    assert heavy == sorted(heavy)


def test_the_unmeasured_first_tiers_exist_and_are_unique_and_disjoint():
    fast, slow = night_run.UNMEASURED_REGRESSION, night_run.UNMEASURED_REGRESSION_SLOW
    assert len(set(fast)) == len(fast)
    assert len(set(slow)) == len(slow)
    assert len(set(night_run.UNMEASURED_TDI)) == len(night_run.UNMEASURED_TDI)
    assert set(fast) <= set(night_run.REGRESSION)
    assert set(slow) <= set(night_run.REGRESSION)
    assert set(night_run.UNMEASURED_TDI) <= set(night_run.TDI)
    assert not set(fast) & set(slow)


def _dense_block_flags(overrides: dict) -> tuple[bool, bool, bool]:
    features = _resolved(overrides).features
    return (
        features.use_physicochemical,
        features.use_mechanistic,
        features.use_ionization_geometry,
    )


@pytest.mark.parametrize("base_name", ["graph", "graph_octant_tox21"])
def test_all_eight_dense_block_combinations_are_declared_for_the_graph(base_name):
    entries = {
        name: overrides
        for name, overrides in night_run.REGRESSION.items()
        if name == base_name
        or name.startswith(f"{base_name}_fs_")
        or (base_name == "graph" and name == "graph_ionization")
    }
    flags = [_dense_block_flags(overrides) for overrides in entries.values()]
    assert len(flags) == len(set(flags))  # no combination twice
    assert set(flags) == set(itertools.product((False, True), repeat=3))


def test_the_tabular_and_neural_families_get_the_full_feature_set_panel():
    for prefix in ("tfm_fs", "boosted_fs", "neural_fs"):
        assert len([n for n in night_run.REGRESSION if n.startswith(f"{prefix}_")]) >= 6
    assert len([n for n in night_run.TDI if n.startswith("curve_fs_")]) >= 6
    assert len([n for n in night_run.TDI if n.startswith("curve_graph_heads_fs_")]) == 7


def test_each_enzyme_descriptor_source_is_swept_alone_and_all_together():
    from cypshift.data.conditioning import DESCRIPTOR_SOURCES

    for pool, template, field in (
        (night_run.REGRESSION, "neural_cond_descriptor_{}", "conditioning.descriptor_sources"),
        (night_run.TDI, "curve_descriptor_{}", "model.curve.descriptor_sources"),
        (night_run.TDI, "curve_graph_descriptor_{}", "model.curve.descriptor_sources"),
    ):
        swept = {
            template.format(s): pool[template.format(s)][field] for s in night_run.DESCRIPTOR_SETS
        }
        alone = {sources[0] for sources in swept.values() if len(sources) == 1}
        assert alone == set(DESCRIPTOR_SOURCES) - {"curated"}
        assert set(swept[template.format("all")]) == set(DESCRIPTOR_SOURCES)


def test_the_unmeasured_first_tiers_are_in_cost_order():
    def kind(name):
        return night_run.REGRESSION[name].get("model.kind", "boosted")

    fast, slow = night_run.UNMEASURED_REGRESSION, night_run.UNMEASURED_REGRESSION_SLOW
    kinds = [kind(n) for n in fast]
    assert set(kinds) == {"tfm", "neural", "graph", "curve"}
    assert max(i for i, k in enumerate(kinds) if k in ("tfm", "neural")) < min(
        i for i, k in enumerate(kinds) if k == "graph"
    )
    assert not any(night_run.REGRESSION[n].get("sources.veith") for n in fast)
    assert all(night_run.REGRESSION[n].get("sources.veith") for n in slow if kind(n) == "graph")
    boosted = [i for i, n in enumerate(slow) if kind(n) == "boosted"]
    assert boosted == list(range(len(slow) - len(boosted), len(slow)))  # boosted last
    tdi_graph = [n for n in night_run.UNMEASURED_TDI if "graph" in n]
    assert night_run.UNMEASURED_TDI[-len(tdi_graph) :] == tuple(tdi_graph)  # MLP curves first


def _grid_cells() -> dict[tuple[str, str, str], dict]:
    cells = {}
    for init, screen, readout in itertools.product(
        ("", "_chemeleon"), ("", "_log2fc"), ("", "_measurement_heads")
    ):
        name = f"graph_panel{init}{screen}{readout}"
        cells[(init, screen, readout)] = night_run.REGRESSION[name]
    return cells


def test_the_panel_grid_is_a_full_factorial_that_varies_one_factor_per_axis():
    cells = _grid_cells()
    assert len(cells) == 8
    varied = {
        key
        for key in {k for cell in cells.values() for k in cell}
        if len({repr(cell.get(key)) for cell in cells.values()}) > 1
    }
    assert varied == {
        "model.graph.pretraining",
        "multifidelity.screen_strategy",
        "model.graph.per_measurement_heads",
    }
    for cell in cells.values():  # every cell keeps the screen and both Veith quantities
        resolved = _resolved(cell)
        assert resolved.sources.veith and resolved.sources.qhts_representation == "both"
        assert resolved.model.graph.ffn_hidden == 300  # readout width pinned across the readouts
        assert resolved.multifidelity.screen_strategy in ("response_head", "log2fc_response")


def test_the_screen_off_controls_differ_from_their_panel_cells_only_in_the_screen():
    for control, cell in (
        ("graph_panel_no_screen", "graph_panel"),
        ("graph_panel_chemeleon_no_screen", "graph_panel_chemeleon"),
    ):
        off, on = _resolved(night_run.REGRESSION[control]), _resolved(night_run.REGRESSION[cell])
        assert not off.sources.screen and on.sources.screen
        assert off.sources.veith == on.sources.veith
        assert off.model.graph.pretraining == on.model.graph.pretraining
        assert off.model.graph.ffn_hidden == on.model.graph.ffn_hidden


def test_the_combined_chembl_recipe_is_the_grid_corner_plus_chembl_only():
    corner = _resolved(night_run.REGRESSION["graph_panel_chemeleon_log2fc_measurement_heads"])
    for name in (
        "graph_panel_chemeleon_log2fc_measurement_heads_chembl",
        "graph_panel_chemeleon_log2fc_measurement_heads_chembl_budget_half",
    ):
        combined = _resolved(night_run.REGRESSION[name])
        assert combined.sources.chembl and not corner.sources.chembl
        assert combined.model.graph.per_measurement_heads
        assert combined.model.graph.pretraining == "chemeleon"
        assert combined.multifidelity.screen_strategy == "log2fc_response"


def test_the_learning_rate_sweep_brackets_the_default_for_each_base():
    default = ExperimentConfig().model.graph.learning_rate
    assert default == 0.0  # chemprop's own 1e-3 peak; the entries below lower it
    for base in ("graph", "graph_chemeleon", "graph_octant_tox21"):
        rates = {
            _resolved(night_run.REGRESSION[f"{base}_learning_rate{tag}"]).model.graph.learning_rate
            for tag in ("0p0001", "0p0003")
        }
        assert rates == {1e-4, 3e-4}


def test_main_runs_the_unmeasured_entries_first_and_writes_the_regression_candidate_before_tdi(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "data" / "night_results.csv")
    monkeypatch.setattr(night_run, "_archive_stale_results", lambda: None)
    monkeypatch.setattr(night_run, "wandb_start", lambda: None)
    monkeypatch.setattr(night_run, "wandb_finish", lambda: None)
    monkeypatch.setattr(night_run, "load_local_env", lambda: [])
    monkeypatch.setattr(
        night_run, "run_phase", lambda phase, configs, seeds: calls.append((phase, tuple(configs)))
    )
    monkeypatch.setattr(
        night_run, "ranked_means", lambda phase: pd.DataFrame({"mean": [0.7]}, index=["graph"])
    )
    monkeypatch.setattr(
        night_run, "ranked_mcc", lambda phase: pd.DataFrame({"mean": [0.5]}, index=["curve_heads"])
    )
    monkeypatch.setattr(
        night_run, "write_submission", lambda track, *args, **kwargs: calls.append(("write", track))
    )

    night_run.main()

    assert calls == [
        ("reg", night_run.DATA_SIDE_REGRESSION),
        ("reg", night_run.QHTS_REGRESSION),
        ("tdi", night_run.QHTS_TDI),
        ("reg", night_run.UNMEASURED_REGRESSION),
        ("tdi", night_run.UNMEASURED_TDI),
        ("reg", night_run.UNMEASURED_REGRESSION_SLOW),
        ("reg", tuple(night_run.REGRESSION)),
        ("write", "regression"),
        ("tdi", tuple(night_run.TDI)),
        ("write", "classification"),
    ]
