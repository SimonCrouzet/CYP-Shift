"""The declared sweep: every entry resolves and validates, the 2026-10-24 review's comparison grid
is a clean factorial with matched controls, the priority entries exist, and the run order puts
them (and the regression candidate) first."""

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


def test_the_unmeasured_first_tiers_exist_and_are_unique_and_disjoint():
    fast, slow = night_run.UNMEASURED_REGRESSION, night_run.UNMEASURED_REGRESSION_SLOW
    assert len(set(fast)) == len(fast)
    assert len(set(slow)) == len(slow)
    assert len(set(night_run.UNMEASURED_TDI)) == len(night_run.UNMEASURED_TDI)
    assert set(fast) <= set(night_run.REGRESSION)
    assert set(slow) <= set(night_run.REGRESSION)
    assert set(night_run.UNMEASURED_TDI) <= set(night_run.TDI)
    assert not set(fast) & set(slow)


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
        ("reg", night_run.UNMEASURED_REGRESSION),
        ("tdi", night_run.UNMEASURED_TDI),
        ("reg", night_run.UNMEASURED_REGRESSION_SLOW),
        ("reg", tuple(night_run.REGRESSION)),
        ("write", "regression"),
        ("tdi", tuple(night_run.TDI)),
        ("write", "classification"),
    ]
