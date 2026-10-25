"""The declared sweep: every entry resolves and validates, the 2026-10-24 review's comparison grid
is a clean factorial with matched controls, the priority entries exist, and the run order puts
them (and the regression candidate) first."""

import dataclasses
import itertools
import json
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


def _greedy_world(monkeypatch, tmp_path, score):
    """Run ``run_greedy`` against a fake measurement: ``score(overrides)`` is a config's macro
    ST-RAE, or None for one that failed. Returns the list of batches that were run."""
    ran: list[dict] = []
    monkeypatch.setattr(night_run, "REGRESSION", dict(night_run.REGRESSION))  # drop new names
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(
        night_run, "run_phase", lambda phase, configs, seeds: ran.append(dict(configs))
    )

    def ranked(phase):
        names = {name: cfg for batch in ran for name, cfg in batch.items()}
        scores = {name: score(cfg) for name, cfg in names.items()}
        return pd.DataFrame({"mean": {n: v for n, v in scores.items() if v is not None}})

    monkeypatch.setattr(night_run, "ranked_means", ranked)
    return ran


_CHEMBL_THEN_REACH = (
    night_run.GreedyStep("sources.chembl", (True,)),
    night_run.GreedyStep(
        "auxiliary.max_distance", (0.0, 3.0), requires=(("sources.chembl", True),)
    ),
)


def test_greedy_adopts_a_clear_winner_and_runs_the_next_step_on_it(monkeypatch, tmp_path):
    def score(cfg):
        base = 0.70 - (0.05 if cfg.get("sources.chembl") else 0.0)
        return base - {0.0: 0.01, 3.0: 0.04}.get(cfg.get("auxiliary.max_distance"), 0.0)

    ran = _greedy_world(monkeypatch, tmp_path, score)
    final = night_run.run_greedy(night_run.DATA_BASE, _CHEMBL_THEN_REACH, margin=0.005)
    assert final["sources.chembl"] is True and final["auxiliary.max_distance"] == 3.0
    reach_batch = ran[-1].values()
    assert all(cfg["sources.chembl"] is True for cfg in reach_batch)  # run on the adopted recipe
    path = json.loads((tmp_path / "greedy_path.json").read_text())
    assert [entry["step"] for entry in path["path"]] == [
        "base",
        "sources.chembl",
        "auxiliary.max_distance",
    ]


def test_greedy_keeps_the_recipe_when_the_gap_is_within_the_margin(monkeypatch, tmp_path):
    ran = _greedy_world(
        monkeypatch, tmp_path, lambda cfg: 0.70 - (0.003 if cfg.get("sources.chembl") else 0.0)
    )
    final = night_run.run_greedy(night_run.DATA_BASE, _CHEMBL_THEN_REACH, margin=0.005)
    assert final == night_run.REGRESSION[night_run.DATA_BASE]
    assert len(ran) == 2  # the base, then the chembl step; the reach step was skipped


def test_greedy_skips_a_step_whose_precondition_is_not_met(monkeypatch, tmp_path):
    ran = _greedy_world(monkeypatch, tmp_path, lambda cfg: 0.70)
    night_run.run_greedy(night_run.DATA_BASE, _CHEMBL_THEN_REACH[1:], margin=0.005)
    assert len(ran) == 1  # only the base ran


def test_greedy_never_adopts_a_configuration_that_did_not_score(monkeypatch, tmp_path):
    _greedy_world(monkeypatch, tmp_path, lambda cfg: None if cfg.get("sources.chembl") else 0.70)
    final = night_run.run_greedy(night_run.DATA_BASE, _CHEMBL_THEN_REACH, margin=0.005)
    assert "sources.chembl" not in final


def test_greedy_reuses_the_name_of_a_recipe_that_is_already_declared(monkeypatch, tmp_path):
    ran = _greedy_world(monkeypatch, tmp_path, lambda cfg: 0.70)
    step = night_run.GreedyStep("sources.screen", (False,))
    night_run.run_greedy(night_run.DATA_BASE, (step,), margin=0.005)
    declared = [n for n, c in night_run.REGRESSION.items() if c.get("sources.screen") is False]
    assert set(ran[-1]) <= set(declared)
    assert not any(name.startswith("greedy_") for batch in ran for name in batch)


def test_greedy_step_sets_its_implied_source_instead_of_waiting_for_it(monkeypatch, tmp_path):
    step = night_run.GreedyStep(
        "sources.qhts_representation", ("max_response",), implies=(("sources.veith", True),)
    )
    ran = _greedy_world(
        monkeypatch,
        tmp_path,
        lambda cfg: 0.70 - (0.02 if cfg.get("sources.qhts_representation") else 0.0),
    )
    final = night_run.run_greedy(night_run.DATA_BASE, (step,), margin=0.005)
    (candidate,) = ran[-1].values()
    assert candidate["sources.veith"] is True  # the base has no Veith; the step turned it on
    assert candidate["sources.qhts_representation"] == "max_response"
    assert final["sources.veith"] is True and final["sources.qhts_representation"] == "max_response"
    path = json.loads((tmp_path / "greedy_path.json").read_text())
    assert path["recipe"]["sources.veith"] is True


def test_the_qhts_representation_is_tried_on_the_recipes_own_qhts_sources_then_with_veith():
    steps = [s for s in night_run.GREEDY_STEPS if s.key == "sources.qhts_representation"]
    assert [s.implies for s in steps] == [(), (("sources.veith", True),)]
    assert all(s.values == ("max_response", "both") for s in steps)


def test_no_greedy_step_waits_on_the_veith_source_being_adopted():
    steps = {step.key: step for step in night_run.GREEDY_STEPS}
    assert ("sources.veith", True) in steps["auxiliary.include_qhts_inactives"].implies
    assert not any(("sources.veith", True) in step.requires for step in night_run.GREEDY_STEPS)


def test_every_qhts_greedy_candidate_is_a_valid_configuration_without_prior_veith():
    base = dict(night_run.REGRESSION[night_run.DATA_BASE])
    assert not base.get("sources.veith")
    for step in night_run.GREEDY_STEPS:
        if step.key not in ("sources.qhts_representation", "auxiliary.include_qhts_inactives"):
            continue
        key = step.key
        for value in step.values:
            config = ExperimentConfig()
            for field, setting in {**base, **dict(step.implies), key: value}.items():
                config.override(field, setting)
            config.validate()


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
        night_run, "run_greedy", lambda base, steps, *a, **k: calls.append(("greedy", base))
    )
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
        ("reg", night_run.QHTS_REGRESSION),
        ("tdi", night_run.QHTS_TDI),
        ("greedy", night_run.DATA_BASE),
        ("reg", night_run.DATA_SIDE_REGRESSION),
        ("reg", night_run.UNMEASURED_REGRESSION),
        ("tdi", night_run.UNMEASURED_TDI),
        ("reg", night_run.UNMEASURED_REGRESSION_SLOW),
        ("reg", tuple(night_run.REGRESSION)),
        ("write", "regression"),
        ("tdi", tuple(night_run.TDI)),
        ("write", "classification"),
    ]


def test_the_queue_starts_with_the_max_response_on_the_best_recipe():
    first = night_run.QHTS_REGRESSION[0]
    recipe = night_run.REGRESSION[first]
    assert recipe["sources.qhts_representation"] == "max_response"
    assert recipe["model.kind"] == "graph"
    assert recipe["sources.novartis"] and recipe["sources.octant"]  # the 0.6415 recipe


def test_the_qhts_cells_cover_every_source_that_supplies_a_max_response():
    from cypshift.data.sources import supplying

    suppliers = supplying("max_response")
    assert len(suppliers) >= 2
    cells = night_run.QHTS_VARIANTS
    for name in suppliers:
        assert f"{name}_max_response" in cells and f"{name}_both" in cells
    assert "_".join(suppliers) + "_max_response" in cells  # all of them together
