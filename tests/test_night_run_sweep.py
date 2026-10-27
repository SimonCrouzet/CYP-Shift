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
    "sources.tdi_label": "classification track; varied by the TDI cells of the questions plan",
    "multifidelity.screen_model_hidden": "screen helper width; the graph cannot take that helper",
    "training.preset": "curriculum; the graph ignores stage settings",
    "training.stages": "curriculum; the graph ignores stage settings",
    "auxiliary.row_selection": "varied by the subset plan's cells, registered when it runs",
    "auxiliary.max_rows_per_group": "varied by the subset plan's cells, registered when it runs",
    "auxiliary.row_share": "varied by the share plan's cells, registered when it runs",
    "auxiliary.row_selection_assays": "varied by the share plan's cells, registered when it runs",
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
        night_run, "run_phase", lambda phase, configs, seeds, extra=None: ran.append(dict(configs))
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
        ("reg", night_run.PANEL_RERUN_REGRESSION),
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


def test_the_sign_bugged_panel_cells_rerun_and_their_old_rows_are_never_carried_over():
    assert len(set(night_run.PANEL_RERUN_REGRESSION)) == 6
    assert all(not rev.endswith("-signbug") for rev in [night_run.REV, *night_run.CARRIED])


# --- the focus plan ----------------------------------------------------------------------


def _focus_world(monkeypatch, tmp_path, score):
    """Run the focus plan against a fake measurement: ``score(overrides)`` is a recipe's macro
    ST-RAE, or None for one that failed. Returns the batches that were run, in order."""
    ran: list[dict] = []
    monkeypatch.setattr(night_run, "REGRESSION", dict(night_run.REGRESSION))
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(
        night_run, "run_phase", lambda phase, configs, seeds, extra=None: ran.append(dict(configs))
    )

    def ranked(phase):
        names = {name: cfg for batch in ran for name, cfg in batch.items()}
        scores = {name: score(cfg) for name, cfg in names.items()}
        return pd.DataFrame({"mean": {n: v for n, v in scores.items() if v is not None}})

    monkeypatch.setattr(night_run, "ranked_means", ranked)
    monkeypatch.setattr(night_run, "blend_report", lambda final, names: pd.DataFrame())
    return ran


def _base_score(cfg):
    return 0.64


def test_the_focus_cells_are_valid_configurations_on_every_recipe_they_can_meet():
    base = night_run.REGRESSION[night_run.FOCUS_BASE]
    round_2_winners = [base] + [
        night_run._apply(base, cell.sets) for cell in night_run.FOCUS_ROUND_1
    ]
    for recipe in round_2_winners:
        for cell in night_run.FOCUS_ROUND_2:
            assert night_run._is_valid(night_run._apply(recipe, cell.sets)), cell.tag
    recipes = [
        night_run._apply(winner, cell.sets)
        for winner in round_2_winners
        for cell in (None, *night_run.FOCUS_ROUND_2)
        if cell is not None
    ] + round_2_winners
    for recipe in recipes:
        for cell in night_run.FOCUS_ROUND_3:
            # a cell that does not fit the incumbent is skipped by the plan, never run invalid
            overrides = night_run._apply(recipe, cell.sets)
            assert isinstance(night_run._is_valid(overrides), bool)
    for cell in night_run.FOCUS_ROUND_3:
        assert night_run._is_valid(night_run._apply(base, cell.sets)), cell.tag


def test_round_1_crosses_the_qhts_representation_with_the_weights_and_keeps_the_baseline():
    base = night_run.REGRESSION[night_run.FOCUS_BASE]
    assert base.get("sources.qhts_representation", "pic50") == "pic50"
    assert base.get("auxiliary.per_endpoint_budget", 0.0) == 0.0
    cells = {c.tag: dict(c.sets) for c in night_run.FOCUS_ROUND_1}
    assert cells["pic50_budget"] == {"auxiliary.per_endpoint_budget": night_run.FOCUS_BUDGET}
    assert cells["both"] == {"sources.qhts_representation": "both"}
    assert cells["both_budget"] == {**cells["both"], **cells["pic50_budget"]}
    assert base["sources.novartis"] is True  # Novartis is in every cell, not only the baseline


def test_round_2_crosses_the_screen_representation_with_the_readout():
    cells = {c.tag: dict(c.sets) for c in night_run.FOCUS_ROUND_2}
    assert cells["log2fc"] == {"multifidelity.screen_strategy": "log2fc_response"}
    assert cells["mheads"]["model.graph.per_measurement_heads"] is True
    assert cells["log2fc_mheads"] == {**cells["log2fc"], **cells["mheads"]}
    assert cells["mheads"]["model.graph.ffn_hidden"] == 300  # same readout width as the shared one


def test_the_model_cells_cover_the_six_experiments_of_the_model_side_plan():
    cells = {c.tag: dict(c.sets) for c in night_run.FOCUS_ROUND_3}
    assert cells["epochs60"] == {"model.graph.epochs": 60}
    assert cells["capacity"] == {
        "model.graph.hidden": 600,
        "model.graph.depth": 5,
        "model.graph.ffn_hidden": 1024,
        "model.graph.epochs": 60,
    }
    assert cells["per_target_heads"]["model.graph.per_target_heads"] is True
    assert cells["per_measurement_heads"]["model.graph.per_measurement_heads"] is True
    assert cells["chemeleon_lr3e4"] == {
        "model.graph.pretraining": "chemeleon",
        "model.graph.learning_rate": 3e-4,
        "model.graph.epochs": 60,
    }
    assert cells["point_loss0p1"] == {"model.point_loss_weight": 0.1}
    assert cells["absolute"] == {"model.objective": "absolute"}
    assert cells["adapter_cyp2d6"] == {"model.graph.adapt_target": "CYP2D6"}


def test_the_focus_plan_adopts_each_rounds_winner_and_runs_the_next_round_on_it(
    monkeypatch, tmp_path
):
    def score(cfg):
        both = cfg.get("sources.qhts_representation") == "both"
        mheads = cfg.get("model.graph.per_measurement_heads", False)
        return 0.64 - (0.02 if both else 0.0) + (0.05 if mheads else 0.0)

    ran = _focus_world(monkeypatch, tmp_path, score)
    final = night_run.run_focus(margin=0.005)
    assert len(ran[0]) == 4 and night_run.FOCUS_BASE in ran[0]  # the baseline and three cells
    assert all(c.get("sources.qhts_representation") == "both" for c in ran[1].values())
    assert final["sources.qhts_representation"] == "both"
    assert not final.get("model.graph.per_measurement_heads", False)
    path = json.loads((tmp_path / "focus_path.json").read_text())
    assert [entry["round"] for entry in path["path"]] == ["base", "1 data", "2 data", "3 model"]
    assert json.loads((tmp_path / "focus_best.json").read_text())["name"]


def test_the_focus_plan_keeps_the_baseline_when_nothing_beats_it_by_the_margin(
    monkeypatch, tmp_path
):
    ran = _focus_world(monkeypatch, tmp_path, lambda cfg: 0.64 - 0.003)
    final = night_run.run_focus(margin=0.005)
    assert final == night_run.REGRESSION[night_run.FOCUS_BASE]
    assert len(ran) == 3  # one batch per round, none adopted


def test_the_model_round_stacks_the_winning_groups_and_measures_the_stack(monkeypatch, tmp_path):
    def score(cfg):
        return (
            0.64
            - (0.02 if cfg.get("model.graph.epochs") == 60 else 0.0)
            - (0.02 if cfg.get("model.objective") == "absolute" else 0.0)
            - (0.02 if cfg.get("model.graph.adapt_target") == "CYP2D6" else 0.0)
        )

    ran = _focus_world(monkeypatch, tmp_path, score)
    final = night_run.run_focus(margin=0.005)
    assert final["model.graph.epochs"] == 60
    assert final["model.objective"] == "absolute"
    assert final["model.graph.adapt_target"] == "CYP2D6"
    assert len(ran) == 4  # three rounds and the stack
    assert [dict(c.sets) for c in night_run.FOCUS_ROUND_3 if c.tag == "epochs60"]


def test_groups_that_cannot_be_stacked_are_not(monkeypatch, tmp_path):
    def score(cfg):
        encoder = cfg.get("model.graph.pretraining") == "chemeleon"
        capacity = cfg.get("model.graph.hidden") == 600
        return 0.64 - (0.03 if encoder else 0.0) - (0.02 if capacity else 0.0)

    _focus_world(monkeypatch, tmp_path, score)
    final = night_run.run_focus(margin=0.005)
    assert final["model.graph.pretraining"] == "chemeleon"
    assert final.get("model.graph.hidden", 300) == 300  # capacity lost to the encoder


def test_the_adapter_does_not_stack_with_an_independent_readout(monkeypatch, tmp_path):
    def score(cfg):
        heads = cfg.get("model.graph.per_target_heads", False)
        adapter = cfg.get("model.graph.adapt_target") == "CYP2D6"
        return 0.64 - (0.03 if heads else 0.0) - (0.02 if adapter else 0.0)

    _focus_world(monkeypatch, tmp_path, score)
    final = night_run.run_focus(margin=0.005)
    assert final["model.graph.per_target_heads"] is True
    assert final.get("model.graph.adapt_target", "none") == "none"
    assert night_run._is_valid(final)


def test_the_pretrained_encoder_is_also_tried_on_the_other_round_1_recipe(monkeypatch, tmp_path):
    def score(cfg):
        return 0.64 - (0.02 if cfg.get("sources.qhts_representation") == "both" else 0.0)

    ran = _focus_world(monkeypatch, tmp_path, score)
    night_run.run_focus(margin=0.005)
    encoders = [
        cfg
        for batch in ran[2:]
        for cfg in batch.values()
        if cfg.get("model.graph.pretraining") == "chemeleon"
    ]
    kinds = {cfg.get("sources.qhts_representation", "pic50") for cfg in encoders}
    assert kinds == {"both", "pic50"}


def test_a_failed_cell_is_never_adopted(monkeypatch, tmp_path):
    _focus_world(
        monkeypatch,
        tmp_path,
        lambda cfg: None if cfg.get("sources.qhts_representation") == "both" else 0.64,
    )
    final = night_run.run_focus(margin=0.005)
    assert final.get("sources.qhts_representation", "pic50") == "pic50"


# --- running configurations concurrently ----------------------------------------------------


def test_run_phase_runs_the_pending_configurations_concurrently_when_jobs_allow(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(night_run, "JOBS", 3)
    monkeypatch.setattr(night_run, "already_done", lambda: {("reg", "a", 0)})
    started = []
    monkeypatch.setattr(night_run, "_run_parallel", lambda phase, todo: started.append(todo))
    night_run.run_phase("reg", {"a": {}, "b": {"x": 1}, "c": {}}, [0])
    assert started == [[("b", {"x": 1}, 0), ("c", {}, 0)]]  # a is recorded already


def test_run_phase_stays_in_process_for_one_job_or_one_pending_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(night_run, "already_done", lambda: set())
    monkeypatch.setattr(night_run, "_run_parallel", lambda *a: pytest.fail("no pool expected"))
    evaluated = []
    monkeypatch.setattr(night_run, "evaluate", lambda *args: evaluated.append(args[1]))
    monkeypatch.setattr(night_run, "JOBS", 1)
    night_run.run_phase("reg", {"a": {}, "b": {}}, [0])
    monkeypatch.setattr(night_run, "JOBS", 4)
    night_run.run_phase("reg", {"c": {}}, [0])
    assert evaluated == ["a", "b", "c"]


def _fake_worker(phase, name, overrides, seed):
    if name == "bad":
        raise RuntimeError("boom")
    return {
        "rev": "r", "phase": phase, "name": name, "seed": seed, "status": "ok",
        "macro_ST_RAE": 0.5, "minutes": 0.0,
    }  # fmt: skip


def test_the_pool_records_each_finished_row_once_and_a_raising_configuration_as_failed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(night_run, "JOBS", 2)
    monkeypatch.setattr(night_run, "_compute_row_in_child", _fake_worker)
    rows = pd.read_csv  # noqa: F841 -- the table is read back below
    night_run._run_parallel("reg", [("ok1", {}, 0), ("bad", {}, 0), ("ok2", {}, 0)])
    table = pd.read_csv(tmp_path / "night_results.csv")
    assert sorted(table["name"]) == ["bad", "ok1", "ok2"]
    assert table.set_index("name").loc["bad", "status"] == "failed"
    assert (table["status"] == "ok").sum() == 2


def test_the_slowest_model_cells_are_submitted_first(monkeypatch, tmp_path):
    ran = _focus_world(monkeypatch, tmp_path, lambda cfg: 0.64)
    night_run.run_focus(margin=0.005)
    ranks = [night_run._fit_cost_rank(cfg) for cfg in list(ran[2].values())[1:]]  # skip incumbent
    assert ranks == sorted(ranks) and ranks[0] == 0


def test_the_submit_plan_refits_the_focus_recipe_without_its_name(monkeypatch, tmp_path):
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    (tmp_path / "focus_best.json").write_text(
        json.dumps({"name": "focus_x", "model.kind": "graph", "model.graph.epochs": 60})
    )
    written = []
    monkeypatch.setattr(night_run, "write_submission", lambda *args: written.append(args))
    night_run.run_focus_submission()
    ((track, recipe, output, config_path),) = written
    assert track == "regression" and recipe == {"model.kind": "graph", "model.graph.epochs": 60}
    assert output == night_run.FOCUS_SUBMISSION and config_path.parent == tmp_path


def test_a_response_cell_that_beats_its_matched_control_is_adopted_below_the_margin(
    monkeypatch, tmp_path
):
    def score(cfg):
        both = cfg.get("sources.qhts_representation") == "both"
        budget = cfg.get("auxiliary.per_endpoint_budget", 0.0) > 0
        return {(False, False): 0.6415, (False, True): 0.6492, (True, False): 0.6415}.get(
            (both, budget), 0.6369
        )

    ran = _focus_world(monkeypatch, tmp_path, score)
    night_run.run_focus(margin=0.005)
    round_2 = list(ran[1].values())
    assert all(c["sources.qhts_representation"] == "both" for c in round_2)
    assert all(c["auxiliary.per_endpoint_budget"] == night_run.FOCUS_BUDGET for c in round_2)


def test_a_response_cell_is_not_adopted_on_its_control_alone_when_it_is_not_the_lowest(
    monkeypatch, tmp_path
):
    def score(cfg):
        both = cfg.get("sources.qhts_representation") == "both"
        budget = cfg.get("auxiliary.per_endpoint_budget", 0.0) > 0
        if budget and not both:
            return 0.70
        return 0.6415 if not budget else 0.65  # the control gap is large, but the baseline is lower

    ran = _focus_world(monkeypatch, tmp_path, score)
    night_run.run_focus(margin=0.005)
    assert all(c.get("sources.qhts_representation", "pic50") == "pic50" for c in ran[1].values())


def test_the_wide_plan_runs_the_focus_recipe_and_four_valid_members(monkeypatch, tmp_path):
    ran = _greedy_world(monkeypatch, tmp_path, lambda cfg: 0.64)
    monkeypatch.setattr(night_run, "blend_report", lambda *args: None)
    night_run.run_wide()
    (batch,) = ran
    incumbent, *members = batch.items()
    assert incumbent[0] == "focus_both_budget_1217e0"
    assert incumbent[1]["sources.qhts_representation"] == "both"
    assert incumbent[1]["auxiliary.per_endpoint_budget"] == night_run.FOCUS_BUDGET
    assert len(members) == len(night_run.WIDE_CELLS) == 4
    assert all(night_run._is_valid(cfg) for _, cfg in members)
    wide = [cfg for _, cfg in members if cfg.get("model.graph.hidden") == 2048]
    assert len(wide) == 3 and all(cfg["model.graph.ffn_hidden"] == 1024 for cfg in wide)
    absolute = [cfg for _, cfg in members if cfg.get("model.objective") == "absolute"]
    assert len(absolute) == 1 and absolute[0]["training.inverse_variance_weighting"] is True
    narrow = [cfg for _, cfg in members if "model.graph.hidden" not in cfg]
    assert len(narrow) == 1 and narrow[0]["training.inverse_variance_weighting"] is True
    ranks = [night_run._fit_cost_rank(cfg) for _, cfg in members]
    assert ranks == sorted(ranks)  # the slowest fits are submitted first


def test_the_finetune_plan_runs_three_staged_cells_beside_the_focus_recipe(monkeypatch, tmp_path):
    ran = _greedy_world(monkeypatch, tmp_path, lambda cfg: 0.64)
    monkeypatch.setattr(night_run, "blend_report", lambda *args: None)
    night_run.run_finetune()
    (batch,) = ran
    incumbent, *members = batch.items()
    assert incumbent[0] == "focus_both_budget_1217e0"
    presets = {cfg["training.preset"] for _, cfg in members}
    assert presets == {"direct_finetune", "direct_finetune_frozen", "pooled_continue"}
    assert all(night_run._is_valid(cfg) for _, cfg in members)


def _pruning_worker(phase, name, overrides, seed):
    from cypshift.models.monitor import PrunedError

    if name == "slow":
        raise PrunedError("fold error above the limit")
    return _fake_worker(phase, name, overrides, seed)


def test_the_pool_records_a_pruned_cell_as_pruned_and_a_rerun_skips_it(monkeypatch, tmp_path):
    monkeypatch.setattr(night_run, "RESULTS", tmp_path / "night_results.csv")
    monkeypatch.setattr(night_run, "JOBS", 2)
    monkeypatch.setattr(night_run, "_compute_row_in_child", _pruning_worker)
    night_run._run_parallel("reg", [("ok1", {}, 0), ("slow", {}, 0)])
    table = pd.read_csv(tmp_path / "night_results.csv").set_index("name")
    assert table.loc["slow", "status"] == "pruned" and table.loc["ok1", "status"] == "ok"
    assert ("reg", "slow", 0) in night_run.already_done()
    assert "slow" not in night_run._completed("reg")["name"].tolist()


def test_a_recipe_keeps_its_name_while_fold_monitoring_is_off():
    plain = night_run._resolved_key({"model.kind": "graph"})
    monitored = night_run._resolved_key({"model.kind": "graph", "model.graph.monitor_fold": True})
    assert "early_stopping_patience" not in plain and "early_stopping_patience" in monitored


def test_a_recipe_has_one_name_under_every_matmul_precision():
    keys = {
        night_run._resolved_key({"model.kind": "graph", "model.graph.matmul_precision": value})
        for value in ("highest", "high", "medium")
    }
    assert len(keys) == 1


def test_a_recipe_keeps_its_name_while_auxiliary_row_selection_is_off():
    plain = night_run._resolved_key({"model.kind": "graph"})
    reduced = night_run._resolved_key(
        {
            "model.kind": "graph",
            "auxiliary.row_selection": "clusters",
            "auxiliary.max_rows_per_group": 2000,
        }
    )
    assert "row_selection" not in plain and "row_selection" in reduced


def test_the_subset_plan_compares_each_selection_with_the_full_table_at_matched_updates():
    tags = [cell.tag for cell in night_run.SUBSET_CELLS]
    assert tags == ["full_18_epochs", "random_2000", "maxmin_2000", "clusters_2000"]
    base = night_run._apply(
        dict(night_run.REGRESSION[night_run.SUBSET_BASE]), night_run._BOTH + night_run._BUDGET
    )
    assert "sources.novartis" not in base
    for cell in night_run.SUBSET_CELLS:
        assert night_run._is_valid(night_run._apply(base, cell.sets))


def test_each_share_cell_differs_from_its_reference_in_one_factor():
    base = night_run._apply(
        dict(night_run.REGRESSION[night_run.SUBSET_BASE]), night_run._BOTH + night_run._BUDGET
    )
    sets = {cell.tag: dict(cell.sets) for cell in night_run.SHARE_CELLS}
    sets["base"] = {}
    for cell in night_run.SHARE_CELLS:
        assert night_run._is_valid(night_run._apply(base, cell.sets))
        changed = set(sets[cell.tag]) ^ set(sets[cell.control])
        changed |= {
            k
            for k in set(sets[cell.tag]) & set(sets[cell.control])
            if sets[cell.tag][k] != sets[cell.control][k]
        }
        factor = {"row selection": {"auxiliary.row_selection", "auxiliary.row_share"}}[cell.group]
        assert changed == factor, cell.tag
    keys = {night_run._resolved_key(night_run._apply(base, c.sets)) for c in night_run.SHARE_CELLS}
    assert len(keys) == len(night_run.SHARE_CELLS)
    assert all("model.graph.epochs" not in dict(c.sets) for c in night_run.SHARE_CELLS)


def test_the_by_source_cells_restrict_maxmin_75_to_one_sources_groups():
    base = night_run._apply(
        dict(night_run.REGRESSION[night_run.SUBSET_BASE]), night_run._BOTH + night_run._BUDGET
    )
    maxmin = dict(next(c for c in night_run.SHARE_CELLS if c.tag == "maxmin_75").sets)
    cells = {cell.tag: dict(cell.sets) for cell in night_run.ASSAY_CELLS}
    assert set(cells) == {"maxmin_75_octant", "maxmin_75_tox21", "maxmin_75_novartis"}
    for sets in cells.values():
        assert night_run._is_valid(night_run._apply(base, tuple(sets.items())))
        assert {k: v for k, v in sets.items() if k != "auxiliary.row_selection_assays"} == maxmin
    assert cells["maxmin_75_octant"]["auxiliary.row_selection_assays"] == ["fluorogenic_pic50"]
    assert cells["maxmin_75_novartis"]["auxiliary.row_selection_assays"] == ["reversible_pic50"]
    assert set(cells["maxmin_75_tox21"]["auxiliary.row_selection_assays"]) == {
        "luminescent_ac50",
        "luminescent_max_response",
    }
    keys = {night_run._resolved_key(night_run._apply(base, c.sets)) for c in night_run.ASSAY_CELLS}
    keys.add(night_run._resolved_key(night_run._apply(base, tuple(maxmin.items()))))
    assert len(keys) == 4


def test_the_same_input_comparison_gives_every_family_the_same_features_and_rows():
    configs = night_run.fair_configs(tuple(night_run.FAIR_FAMILIES))
    assert len(configs) == (
        len(night_run.FAIR_FAMILIES) * len(night_run.FAIR_FEATURES) * len(night_run.FAIR_DATA)
    )
    for name, (_, family, features, data, overrides) in configs.items():
        assert night_run._is_valid(overrides), name
        shared = {k: v for k, v in overrides.items() if k.startswith(("features.", "sources."))}
        expected = {
            **{
                k: v
                for k, v in night_run.FAIR_FAMILIES[family][1].items()
                if k.startswith(("features.", "sources."))
            },
            **night_run.FEATURE_SETS[features],
            **night_run.FAIR_DATA[data],
        }
        assert shared == expected, name
    for features in night_run.FAIR_FEATURES:
        for data in night_run.FAIR_DATA:
            seen = {
                f: {k: v for k, v in c[4].items() if k.startswith("features.")}
                for n, c in configs.items()
                for f in [c[1]]
                if c[2] == features and c[3] == data
            }
            assert len({repr(sorted(v.items())) for v in seen.values()}) == 1, (features, data)
    chemeleon = night_run.FEATURE_SETS["chemeleon"]
    assert chemeleon["features.molecular_embedding"] == "chemeleon"
    assert (
        not chemeleon["features.use_fingerprint"] and not chemeleon["features.use_physicochemical"]
    )


def test_a_row_cap_recipe_does_not_carry_the_unused_share_and_assay_fields():
    capped = night_run._resolved_key(
        {
            "model.kind": "graph",
            "auxiliary.row_selection": "random",
            "auxiliary.max_rows_per_group": 2000,
        }
    )
    assert "max_rows_per_group" in capped
    assert "row_share" not in capped and "row_selection_assays" not in capped


def test_the_factor_report_marks_only_changes_beyond_the_adopt_margin(monkeypatch):
    done = pd.DataFrame(
        {
            "name": ["ref", "ref", "ref", "small", "large", "better"],
            "seed": [0, 1, 2, 0, 0, 0],
            "macro_ST_RAE": [0.6652, 0.7500, 0.7500, 0.6671, 0.6752, 0.6552],
        }
    )
    monkeypatch.setattr(night_run, "_completed", lambda phase=None: done)
    cells = (
        night_run.Cell("small", (), "factor", "base"),
        night_run.Cell("large", (), "factor", "base"),
        night_run.Cell("better", (), "factor", "base"),
    )
    names = {"base": "ref", "small": "small", "large": "large", "better": "better"}
    table = night_run._report_factors(cells, names).set_index("cell")
    assert table["beyond_margin"].to_dict() == {"small": False, "large": True, "better": True}
    assert table.loc["better", "change"] < 0 < table.loc["large", "change"]
    assert (
        table.loc["small", "reference_score"] == 0.6652
    )  # the other seeds of the reference are not averaged in


def _night_base(sets=()):
    return night_run._apply(
        night_run._apply(
            dict(night_run.REGRESSION[night_run.SUBSET_BASE]), night_run._BOTH + night_run._BUDGET
        ),
        sets,
    )


def test_the_night_cells_are_valid_and_each_differs_from_its_reference_in_its_own_setting():
    base = _night_base()
    sets = {cell.tag: dict(cell.sets) for cell in night_run.NIGHT_CELLS}
    sets["base"] = {}
    for cell in night_run.NIGHT_CELLS:
        assert night_run._is_valid(night_run._apply(base, cell.sets)), cell.tag
        assert (cell.control or "base") in sets
        if cell.control == "pooled_continue":
            assert set(sets[cell.tag]) == set(sets["pooled_continue"]) == {"training.preset"}
    keys = {night_run._resolved_key(night_run._apply(base, c.sets)) for c in night_run.NIGHT_CELLS}
    assert len(keys) == len(night_run.NIGHT_CELLS)
    for member in night_run.NIGHT_MEMBERS:
        assert night_run._is_valid(night_run._member_recipe(member, base)), member


def test_a_member_takes_the_base_recipes_data_settings_unless_its_family_cannot():
    base = _night_base(night_run.RANDOM_75.sets)
    boosted = night_run._member_recipe("boosted_pinball_t60", base)
    assert boosted["sources.octant"] and boosted["sources.tox21"]
    assert boosted["auxiliary.per_endpoint_budget"] == night_run.FOCUS_BUDGET
    assert boosted["auxiliary.row_selection"] == "random"
    assert boosted["model.objective"] == "pinball"  # its own model settings are kept
    tfm = night_run._member_recipe("tfm_fs_dense", base)
    assert tfm["sources.octant"] and tfm["sources.tox21"]
    assert "sources.qhts_representation" not in tfm  # no response head: "both" is refused
    assert tfm["auxiliary.per_endpoint_budget"] == night_run.FOCUS_BUDGET
    assert night_run._is_valid(tfm)


def test_the_readout_depth_pooling_and_batch_norm_leave_recorded_names_unchanged_at_defaults():
    plain = night_run._resolved_key({"model.kind": "graph"})
    assert all(f not in plain for f in ("ffn_layers", "batch_norm", "aggregation"))
    deeper = night_run._resolved_key({"model.kind": "graph", "model.graph.ffn_layers": 2})
    assert "ffn_layers" in deeper and deeper != plain


def _policy_scores(monkeypatch, overrides: dict[str, float]):
    """Make ``ranked_means`` return the full table's score and the named cells' scores."""
    declared = {night_run._resolved_key(o): n for n, o in night_run.REGRESSION.items()}
    recipe = _night_base()
    scores = {night_run._recipe_name("both_budget", recipe, declared): 0.6631}
    for cell in (night_run.RANDOM_75, *night_run.SHARE_CELLS):
        name = night_run._recipe_name(cell.tag, night_run._apply(recipe, cell.sets), declared)
        scores[name] = overrides.get(cell.tag, 0.70)
    monkeypatch.setattr(
        night_run, "ranked_means", lambda phase=None: pd.DataFrame({"mean": pd.Series(scores)})
    )


def test_the_row_policy_is_the_best_cell_only_when_it_beats_the_full_table_by_the_margin(
    monkeypatch,
):
    _policy_scores(monkeypatch, {"random_75": 0.6577, "clusters_50": 0.6600})
    assert night_run.choose_row_policy().tag == "random_75"
    _policy_scores(monkeypatch, {"maxmin_75": 0.6590, "clusters_50": 0.6640})
    assert night_run.choose_row_policy() is None  # 0.0041 under the full table: within the margin
    _policy_scores(monkeypatch, {"clusters_50": 0.6500, "random_75": 0.6577})
    assert night_run.choose_row_policy().tag == "clusters_50"


def test_run_phase_adds_the_extra_seeds_of_the_named_recipe_to_one_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(night_run, "already_done", lambda: {("reg", "a", 0)})
    monkeypatch.setattr(
        night_run, "evaluate", lambda phase, name, o, seed: calls.append((name, seed))
    )
    monkeypatch.setattr(night_run, "JOBS", 1)
    night_run.run_phase("reg", {"a": {}, "b": {}}, [0], extra_seeds={"a": [1, 2]})
    assert calls == [("a", 1), ("a", 2), ("b", 0)]


def test_each_member_variant_changes_one_data_setting_of_the_declared_boosted_recipe():
    declared = night_run.REGRESSION["boosted_octant_tox21_deep"]
    for cell in night_run.MEMBER_VARIANTS:
        assert cell.control == "boosted_octant_tox21_deep"
        assert night_run._is_valid(night_run._apply(declared, cell.sets)), cell.tag
        changed = {k for k, v in cell.sets if declared.get(k) != v}
        assert changed == {k for k, _ in cell.sets}, cell.tag
    both, budget, policy = (dict(c.sets) for c in night_run.MEMBER_VARIANTS)
    assert set(both) == {"sources.qhts_representation"}
    assert set(budget) == {"auxiliary.per_endpoint_budget"}
    assert set(policy) == {"auxiliary.row_selection", "auxiliary.row_share"}


def test_the_row_policy_can_be_set_from_the_environment(monkeypatch):
    _policy_scores(monkeypatch, {})  # no cell beats the full table
    monkeypatch.setenv("CYP_ROW_POLICY", "random_75")
    assert night_run.choose_row_policy().tag == "random_75"
    monkeypatch.setenv("CYP_ROW_POLICY", "none")
    assert night_run.choose_row_policy() is None


def test_the_questions_plan_cells_change_one_factor_of_the_novartis_recipe():
    recipe = night_run._apply(
        dict(night_run.REGRESSION[night_run.FOCUS_BASE]), night_run._BOTH + night_run._BUDGET
    )
    assert recipe["sources.novartis"] and recipe["sources.octant"] and recipe["sources.tox21"]
    cells = (
        *night_run.ASSAY_CELLS,
        *night_run.QHTS_CELLS,
        *night_run.CHEMELEON_CELLS,
        *night_run.EMBEDDING_CELLS,
        *night_run.CAPACITY_CELLS,
        *night_run.EPOCHS_CELLS,
        *night_run.FINETUNE_30_CELLS,
        *night_run.TDI_GRAPH_CELLS,
    )
    sets = {cell.tag: dict(cell.sets) for cell in cells}
    sets["base"] = {}
    capacity = {
        "model.graph.hidden",
        "model.graph.depth",
        "model.graph.ffn_hidden",
        "model.graph.ffn_layers",
        "conditioning.embedding_dim",
    }
    factor = {
        "maxmin_75_octant": {"auxiliary.row_selection", "auxiliary.row_share"},
        "qhts_pic50": {"sources.qhts_representation"},
        "qhts_max_response": {"sources.qhts_representation"},
        "tox21_inactives": {"auxiliary.tox21_include_inactives"},
        "chemeleon": {"model.graph.pretraining"},
        "chemeleon_lr3e4": {"model.graph.learning_rate"},
        "chemeleon_lr3e4_e60": {"model.graph.epochs"},
        "epochs_100": {"model.graph.epochs"},
        "direct_finetune_30": {"training.preset"},
        "tdi_graph_base": {"sources.tdi_assay_condition"},
        "tdi_label_base": {"sources.tdi_label"},
        "tdi_label_condition_base": {"sources.tdi_label", "sources.tdi_assay_condition"},
        "tdi_graph_b_reg": {
            *capacity,
            "model.graph.dropout",
            "model.graph.learning_rate",
            "model.graph.epochs",
            "sources.tdi_assay_condition",
        },
        "graph_b_reg_per_target": {
            *capacity,
            "model.graph.dropout",
            "model.graph.learning_rate",
            "model.graph.epochs",
            "model.graph.finetune_per_target_epochs",
            "model.graph.finetune_per_target_lr",
        },
        "graph_b_reg_per_target_10_lr1e5": {
            *capacity,
            "model.graph.dropout",
            "model.graph.learning_rate",
            "model.graph.epochs",
            "model.graph.finetune_per_target_epochs",
            "model.graph.finetune_per_target_lr",
        },
        "per_target_finetune": {
            "model.graph.finetune_per_target_epochs",
            "model.graph.finetune_per_target_lr",
        },
        "chemeleon_features": {"features.molecular_embedding"},
        "mammal_features": {"features.molecular_embedding"},
        **{
            tag: capacity | extra
            for tag, extra in (
                ("graph_a", set()),
                ("graph_a_drop0p05", {"model.graph.dropout"}),
                ("graph_a_drop0p2", {"model.graph.dropout"}),
                (
                    "graph_b_drop0p25_lr5e4",
                    {"model.graph.dropout", "model.graph.learning_rate", "model.graph.epochs"},
                ),
                (
                    "graph_c_drop0p4_e200",
                    {"model.graph.dropout", "model.graph.learning_rate", "model.graph.epochs"},
                ),
                (
                    "graph_c_reg",
                    {"model.graph.dropout", "model.graph.learning_rate", "model.graph.epochs"},
                ),
                ("graph_b", {"model.graph.dropout"}),
                ("graph_c", {"model.graph.dropout", "model.graph.learning_rate"}),
                (
                    "graph_b_reg",
                    {"model.graph.dropout", "model.graph.learning_rate", "model.graph.epochs"},
                ),
            )
        },
    }
    for cell in cells:
        assert night_run._is_valid(night_run._apply(recipe, cell.sets)), cell.tag
        reference = cell.control or "base"
        assert reference in sets, cell.tag
        changed = {
            k
            for k in set(sets[cell.tag]) | set(sets[reference])
            if sets[cell.tag].get(k) != sets[reference].get(k)
        }
        if cell.tag.startswith("maxmin_75_"):
            assert changed == {*factor["maxmin_75_octant"], "auxiliary.row_selection_assays"}
        else:
            assert changed == factor[cell.tag], cell.tag
    keys = {night_run._resolved_key(night_run._apply(recipe, c.sets)) for c in cells}
    assert len(keys) == len(cells)
    for overrides in night_run.CURVE_CELLS.values():
        assert night_run._is_valid(overrides)
        assert overrides["model.kind"] == "curve" and overrides["sources.novartis"]


def test_the_questions_queue_alternates_graph_and_cpu_fits_and_ends_with_tabpfn(monkeypatch):
    queue = {}
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: queue.update(configs))
    monkeypatch.setattr(night_run, "_report_factors", lambda cells, names: pd.DataFrame())
    monkeypatch.setattr(night_run, "_report_fair", lambda configs: pd.DataFrame())
    monkeypatch.setattr(
        night_run, "ranked_means", lambda phase: pd.DataFrame({"mean": [], "std": [], "count": []})
    )
    night_run.run_questions()
    kinds = [overrides.get("model.kind", "boosted") for overrides in queue.values()]
    assert len(queue) == 51
    assert kinds[-4:] == ["tfm"] * 4 and "tfm" not in kinds[:-4]
    assert kinds[0:6:2] == ["graph"] * 3 and "graph" not in kinds[1:6:2]
    first = list(queue.values())[0]
    assert first.get("model.graph.pretraining") == "chemeleon"  # the slowest graph fits first


def test_shards_of_the_questions_queue_are_disjoint_and_together_complete(monkeypatch):
    def queue(shard):
        seen = {}
        monkeypatch.setenv("CYP_SHARD", shard)
        monkeypatch.setattr(
            night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs)
        )
        night_run.run_questions()
        return seen

    first, second = queue("0/2"), queue("1/2")
    monkeypatch.delenv("CYP_SHARD")
    whole = {}
    monkeypatch.setattr(night_run, "run_phase", lambda p, configs, s: whole.update(configs))
    monkeypatch.setattr(night_run, "_report_factors", lambda cells, names: pd.DataFrame())
    monkeypatch.setattr(night_run, "_report_fair", lambda configs: pd.DataFrame())
    monkeypatch.setattr(
        night_run, "ranked_means", lambda phase: pd.DataFrame({"mean": [], "std": [], "count": []})
    )
    night_run.run_questions()
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(whole)
    for part in (first, second):
        kinds = {overrides.get("model.kind", "boosted") for overrides in part.values()}
        assert "graph" in kinds and "tfm" in kinds
    with pytest.raises(ValueError):
        night_run.shard_of({"a": {}}, "2/2")


def test_the_queue_switch_splits_the_questions_plan_into_graph_and_other_cells(monkeypatch):
    def queue(which):
        seen = {}
        monkeypatch.setenv("CYP_QUEUE", which)
        monkeypatch.setattr(
            night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs)
        )
        night_run.run_questions()
        return seen

    graph, other = queue("graph"), queue("other")
    kinds = {c.get("model.kind", "boosted") for c in graph.values()}
    assert kinds == {"graph"} and len(graph) == 30
    assert "graph" not in {c.get("model.kind", "boosted") for c in other.values()}
    assert len(graph) + len(other) == 51
    monkeypatch.setenv("CYP_QUEUE", "none")
    with pytest.raises(ValueError):
        night_run.run_questions()


def test_the_chemeleon_and_plain_queues_split_the_graph_cells(monkeypatch):
    def queue(which):
        seen = {}
        monkeypatch.setenv("CYP_QUEUE", which)
        monkeypatch.setattr(
            night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs)
        )
        night_run.run_questions()
        return seen

    chemeleon, plain, graph = queue("chemeleon"), queue("plain"), queue("graph")
    assert len(chemeleon) == 3 and len(plain) == 13 and len(graph) == 30
    assert not set(chemeleon) & set(plain)
    rest = set(queue("embedding")) | set(queue("capacity"))
    assert set(chemeleon) | set(plain) | rest == set(graph)
    assert len(queue("embedding")) == 2 and len(queue("capacity")) == 12
    assert all(o.get("model.graph.pretraining") == "chemeleon" for o in chemeleon.values())
    assert not any(o.get("model.graph.pretraining") == "chemeleon" for o in plain.values())


def test_the_short_chemeleon_queue_leaves_out_the_sixty_epoch_cell(monkeypatch):
    seen = {}
    monkeypatch.setenv("CYP_QUEUE", "chemeleon30")
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs))
    night_run.run_questions()
    assert len(seen) == 2
    assert all(o["model.graph.pretraining"] == "chemeleon" for o in seen.values())
    assert all(o.get("model.graph.epochs", 30) == 30 for o in seen.values())


def test_named_cells_select_graph_cells_of_the_questions_plan_and_unknown_names_are_refused(
    monkeypatch,
):
    seen = {}
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs))
    monkeypatch.setenv("CYP_CELLS", "qhts_pic50,maxmin_75_novartis")
    night_run.run_questions()
    assert len(seen) == 2
    assert {o.get("sources.qhts_representation") for o in seen.values()} == {"pic50", "both"}
    monkeypatch.setenv("CYP_CELLS", "no_such_cell")
    with pytest.raises(ValueError, match="names no cell"):
        night_run.run_questions()


def test_cyp_seeds_run_the_chosen_cells_at_those_seeds_only(monkeypatch):
    seen = []
    monkeypatch.setenv("CYP_CELLS", "base")
    monkeypatch.setenv("CYP_SEEDS", "1,2")
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: seen.append(seeds))
    night_run.run_questions()
    assert seen == [[1, 2]]


def test_the_boosted_cells_run_on_the_base_recipes_data_and_alone_by_tag(monkeypatch):
    seen = {}
    monkeypatch.setenv("CYP_CELLS", "boosted_deep_base_data,boosted_pinball_base_data")
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs))
    night_run.run_questions()
    deep, pinball = seen.values()
    for overrides in (deep, pinball):
        assert overrides["model.kind"] == "boosted"
        assert overrides["sources.novartis"] is True
        assert overrides["sources.qhts_representation"] == "both"
        assert overrides["auxiliary.per_endpoint_budget"] == night_run.FOCUS_BUDGET
    assert deep["model.boosted.n_estimators"] == 1000
    assert pinball["model.objective"] == "pinball" and pinball["model.objective_tau"] == 0.6


def test_a_curve_cell_can_be_selected_alone_by_its_tag(monkeypatch):
    seen = {}
    monkeypatch.setenv("CYP_CELLS", "curve_heads_both_budget")
    monkeypatch.setattr(night_run, "run_phase", lambda phase, configs, seeds: seen.update(configs))
    night_run.run_questions()
    (overrides,) = seen.values()
    assert overrides["model.kind"] == "curve"
    assert overrides["sources.qhts_representation"] == "both"
