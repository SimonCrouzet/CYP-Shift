"""Turning a configuration into an experiment."""

import numpy as np
import pandas as pd
import pytest

from cypshift.config import ISOFORMS
from cypshift.experiment import ExperimentConfig
from cypshift.pipeline import (
    assemble_training_data,
    build_model,
    metrics_for_tracking,
    weight_by_inverse_variance,
)


def _retained(n=30):
    rng = np.random.default_rng(0)
    frame = {"Molecule_Name": [f"OCNT-{i:04d}" for i in range(n)], "SMILES": ["CCO"] * n}
    for isoform in ISOFORMS:
        frame[f"{isoform}_pIC50_direct_inhibition"] = rng.uniform(3, 7, n)
    return pd.DataFrame(frame)


def test_inverse_variance_weighting_upweights_tight_bands_and_preserves_the_mass():
    """Tier C step 6: a narrower credible interval gets more weight than a wider one, censored
    and response rows are left alone, and the total weight of the affected rows is preserved."""
    obs = pd.DataFrame(
        {
            "value": [5.0, 5.0, 5.0, 0.3],
            "conf_low": [4.9, 4.0, -1e4, np.nan],  # tight, wide, censored(one-sided), response
            "conf_high": [5.1, 6.0, 5.2, np.nan],
            "is_censored": [False, False, True, False],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out = weight_by_inverse_variance(obs)
    assert out.loc[0, "weight"] > out.loc[1, "weight"]  # tight band pulls harder than the wide one
    assert out.loc[2, "weight"] == 1.0  # censored: untouched
    assert out.loc[3, "weight"] == 1.0  # response row (no two-sided interval): untouched
    two_sided = [0, 1]  # the rows the reweighting touched
    assert out.loc[two_sided, "weight"].sum() == pytest.approx(obs.loc[two_sided, "weight"].sum())


def test_inverse_variance_weighting_excludes_response_rows_and_absent_uncertainty():
    """A measured potency interval is eligible; a response-task row (a construction band in
    response units) and a zero-width row (no interval supplied, filled low==high==value) are not,
    so neither collects the maximum-precision factor that would swamp the scored potency rows."""
    from cypshift.data.assembly import SCREEN_RESPONSE_ASSAY

    obs = pd.DataFrame(
        {
            "assay": ["direct_dose_response", SCREEN_RESPONSE_ASSAY, "literature_ic50"],
            "value": [5.0, 0.3, 6.0],
            "conf_low": [4.9, 0.2, 6.0],  # potency band, response band, zero-width potency
            "conf_high": [5.1, 0.4, 6.0],
            "is_censored": [False, False, False],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    out = weight_by_inverse_variance(obs)
    assert out.loc[1, "weight"] == 1.0  # response row: untouched despite its finite band
    assert out.loc[2, "weight"] == 1.0  # zero-width potency: absent uncertainty, untouched
    # Only the single eligible potency row remained; preserving its mass leaves it unchanged.
    assert out.loc[0, "weight"] == pytest.approx(1.0)


def _potency_and_screen_rows():
    """The review's probe: one direct potency with a width-2 interval, one screen fraction with
    the curve model's width-0.2 construction band, and a pseudo-potency from the isotonic map."""
    return pd.DataFrame(
        {
            "assay": ["direct_inhibition_drc", "direct_inhibition_screen"],
            "value": [5.0, 0.3],
            "conf_low": [4.0, 0.2],
            "conf_high": [6.0, 0.4],
            "is_censored": [False, False],
            "weight": [1.0, 0.3],
        }
    )


def test_the_curve_models_screen_rows_are_responses_for_precision_weighting():
    """``direct_inhibition_screen`` is an inhibited fraction under the curve model: reading its
    +-0.1 construction band as a potency interval turned the 0.3 screen/direct weight ratio into
    30 (weights 0.0419 / 1.258 for the probe below). The tag is excluded only for that model."""
    from cypshift.pipeline import precision_response_assays

    curve = ExperimentConfig()
    curve.override("model.kind", "curve")
    out = weight_by_inverse_variance(
        _potency_and_screen_rows(), response_assays=precision_response_assays(curve)
    )
    assert out["weight"].tolist() == pytest.approx([1.0, 0.3])  # screen untouched

    unguarded = weight_by_inverse_variance(_potency_and_screen_rows())
    assert unguarded.loc[1, "weight"] / unguarded.loc[0, "weight"] == pytest.approx(30.0)


@pytest.mark.parametrize("kind", ["boosted", "neural", "graph"])
def test_other_families_keep_precision_weighting_their_screen_pseudo_potencies(kind):
    """Their screen rows are isotonic pIC50 pseudo-labels with a real interval, which the
    weighting handles as potencies: the tag must stay eligible for them."""
    from cypshift.pipeline import precision_response_assays

    config = ExperimentConfig()
    config.override("model.kind", kind)
    assert "direct_inhibition_screen" not in precision_response_assays(config)


def test_the_realized_inventory_is_logged_per_source_assay_and_target(caplog):
    """A run's trained-on data is read off this table, not off the requested flags: a source whose
    loader failed (or an enzyme a cache lacks) shows as a missing row or a zero cell."""
    import logging

    from cypshift.pipeline import log_realized_inventory

    observations = pd.DataFrame(
        {
            "source": ["screen", None, "veith"],
            "assay": [
                "direct_inhibition_screen_response",
                "direct_inhibition_drc",
                "luminescent_ac50",
            ],
            "target": ["CYP1A2", "CYP3A4", "CYP1A2"],
            "weight": [0.3, 1.0, 0.5],
        }
    )
    with caplog.at_level(logging.INFO, logger="cypshift.pipeline"):
        table = log_realized_inventory(observations)
    assert (
        table.loc[("challenge", "direct_inhibition_drc"), "CYP3A4"] == 1
    )  # a missing source -> challenge
    assert table.loc[("veith", "luminescent_ac50"), "CYP3A4"] == 0  # a zero cell is visible
    assert table["weight"].to_dict() == {
        ("challenge", "direct_inhibition_drc"): 1.0,
        ("screen", "direct_inhibition_screen_response"): 0.3,
        ("veith", "luminescent_ac50"): 0.5,
    }
    assert "realized training table: 3 rows" in caplog.text


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


def test_screen_off_uses_only_the_curves():
    config = ExperimentConfig()
    config.sources.screen = False
    observations, screen_model = assemble_training_data(config, _retained())
    assert set(observations["assay"]) == {"direct_inhibition_drc"}
    assert screen_model is None


def test_an_unknown_strategy_says_so_rather_than_falling_back():
    """Declaring an option in the config must not silently do something else."""
    config = ExperimentConfig()
    config.multifidelity.screen_strategy = "distillation"
    with pytest.raises(ValueError, match="unknown screen strategy"):
        assemble_training_data(config, _retained())


@pytest.mark.parametrize("strategy", ["predicted_label", "embedding"])
@pytest.mark.parametrize("kind", ["graph", "curve"])
def test_a_screen_model_strategy_is_refused_where_it_makes_no_sense(kind, strategy):
    """A screen_model predicts what the screen would have said, for a model that cannot
    use the reading itself. The curve model can: the screen is inhibition at 4.95e-5 M,
    a point on the curve it predicts, so substituting a prediction of a measurement we
    hold would be strictly worse. The graph model has no dense matrix to append to."""
    config = ExperimentConfig()
    config.sources.screen = True
    config.multifidelity.screen_strategy = strategy
    config.override("model.kind", kind)
    with pytest.raises(ValueError, match="is not available for"):
        assemble_training_data(config, _retained())


@pytest.mark.parametrize("kind", ["boosted", "neural"])
def test_scalar_models_keep_the_screen_model_strategies(kind):
    """They predict a pIC50 and so cannot consume a response measurement; the screen_model
    is the only route the screen has into them."""
    from cypshift.pipeline import supported_screen_strategy

    assert set(supported_screen_strategy(kind)) >= {"predicted_label", "embedding"}


def test_source_weights_reach_the_observation_table():
    config = ExperimentConfig()
    config.sources.screen = False
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


def _retained_with_tdi(n: int = 40) -> pd.DataFrame:
    """Retained compounds that actually appear in the preincubated file."""
    from cypshift.data.challenge import load

    tdi = load("tdi")
    names = tdi["Molecule_Name"].dropna().unique()[:n]
    rng = np.random.default_rng(0)
    frame = {"Molecule_Name": list(names), "SMILES": ["CCO"] * len(names)}
    for isoform in ISOFORMS:
        frame[f"{isoform}_pIC50_direct_inhibition"] = rng.uniform(3, 7, len(names))
    return pd.DataFrame(frame)


def test_the_preincubated_assay_condition_is_absent_when_all_its_compounds_are_excluded():
    """Excluding every TDI compound (as a holdout/test set would, in the limit) leaves no
    assay_condition, and an empty frame is treated as the source being absent."""
    from cypshift.data.challenge import load

    all_tdi = set(load("tdi")["Molecule_Name"].dropna())
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_assay_condition = True
    observations, _ = assemble_training_data(config, _retained_with_tdi(), exclude_names=all_tdi)
    assert "tdi_drc" not in set(observations["assay"])


def test_the_preincubated_assay_condition_reaches_the_training_table():
    """Omitting it reduced the classification track to a constant "no": the curve
    model's second assay_condition trained on nothing, so predict_is_time_dependent compared an
    untrained assay_condition against a trained one and scored MCC 0.0 -- identical to answering
    "no" to every compound. build_observations accepted a tdi argument that the
    pipeline never passed."""
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_assay_condition = True
    observations, _ = assemble_training_data(config, _retained_with_tdi())
    assert "tdi_drc" in set(observations["assay"])


def test_the_assay_condition_is_off_by_default_to_protect_the_regression_track():
    """It costs about 0.04 macro ST-RAE at every weight tried, more than every gain
    measured so far combined. The classification track turns it on instead."""
    assert ExperimentConfig().sources.tdi_assay_condition is False


def test_the_preincubated_assay_condition_can_be_switched_off():
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_assay_condition = False
    observations, _ = assemble_training_data(config, _retained_with_tdi())
    assert "tdi_drc" not in set(observations["assay"])


def test_the_preincubated_assay_condition_excludes_named_compounds_but_keeps_the_cohort():
    """Held-out and test compounds stay out of the TDI assay_condition (no leakage), but
    the rest of the cohort stays in -- including compounds with a preincubated
    assay_condition but no direct curve, which the old intersection with the direct set
    dropped."""
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_assay_condition = True
    # The synthetic direct set shares no name with the TDI file, yet the assay_condition is
    # still present:
    # the cohort is kept rather than intersected away.
    full, _ = assemble_training_data(config, _retained())
    assay_condition = full[full["assay"] == "tdi_drc"]
    assert not assay_condition.empty
    victim = assay_condition["Molecule_Name"].iloc[0]
    pruned, _ = assemble_training_data(config, _retained(), exclude_names={victim})
    pruned_assay_condition = pruned[pruned["assay"] == "tdi_drc"]
    assert victim not in set(pruned_assay_condition["Molecule_Name"])
    assert len(pruned_assay_condition) < len(assay_condition)


def test_auxiliary_observations_are_weighted_and_appended(monkeypatch):
    """The auxiliary config must actually pull and weight rows -- it was swept but
    unwired. Mocked so the test needs no network."""
    import cypshift.pipeline as pipe
    from cypshift.data.auxiliary import Auxiliary

    monkeypatch.setattr(pipe, "site_distances", lambda scored, cands: None, raising=False)
    monkeypatch.setattr(
        "cypshift.data.auxiliary.select_auxiliaries",
        lambda *a, **k: [Auxiliary("CYP2C19", "CYP2C9", 0.53, 0.6)],
    )
    monkeypatch.setattr("cypshift.data.sequences.sequence_for", lambda name, *a, **k: "M")
    monkeypatch.setattr("cypshift.data.auxiliary.site_distances", lambda scored, cands: None)
    monkeypatch.setattr(
        "cypshift.data.chembl.load_auxiliary",
        lambda enzyme, smiles, *a, **k: pd.DataFrame(
            {"SMILES": ["CCO", "CCN"], "value": [5.0, 6.0], "target": enzyme}
        ),
    )
    config = ExperimentConfig()
    config.sources.chembl = True
    config.auxiliary.target_weight = 1.0
    rows = pipe.auxiliary_observations(config, _retained())
    import numpy as np

    # The relative is admitted and down-weighted; the scored enzymes come in at distance 0.
    assert "CYP2C19" in set(rows["target"])
    relative = rows[rows["target"] == "CYP2C19"]
    assert np.allclose(relative["weight"], 0.6)
    # no invented interval: bounds collapse to the point
    assert (relative["conf_low"] == relative["value"]).all()


def test_response_rows_go_to_a_helper_model_for_models_without_a_response_head(monkeypatch):
    """The graph keeps the max-response rows in its table (response head); boosted, neural and the
    fingerprint curve get them as a separate frame for the helper model and none in the table."""
    import cypshift.pipeline as pipe
    from cypshift.data.assembly import RESPONSE_ASSAY

    monkeypatch.setattr("cypshift.data.sequences.sequence_for", lambda name, *a, **k: "M")
    monkeypatch.setattr("cypshift.data.auxiliary.site_distances", lambda scored, cands: None)
    monkeypatch.setattr("cypshift.data.auxiliary.select_auxiliaries", lambda *a, **k: [])

    def fake_veith(enzyme, smiles, *a, representation="pic50", **k):
        value = 0.4 if representation == "max_response" else 5.0
        return pd.DataFrame({"SMILES": ["CCO", "CCN"], "value": [value] * 2, "target": enzyme})

    monkeypatch.setattr("cypshift.data.veith.load_auxiliary", fake_veith)
    for kind in ("graph", "boosted", "neural", "curve"):
        config = ExperimentConfig()
        config.override("model.kind", kind)
        config.sources.veith = True
        config.override("sources.qhts_representation", "both")
        config.auxiliary.max_distance = 0.0
        table, response = pipe.auxiliary_frames(config, _retained())
        assert (kind != "graph") == pipe.response_feature_route(config)
        assert "luminescent_ac50" in set(table["assay"])
        if kind == "graph":
            assert response.empty and RESPONSE_ASSAY in set(table["assay"])
        else:
            assert RESPONSE_ASSAY not in set(table["assay"])
            assert set(response["assay"]) == {RESPONSE_ASSAY}
            assert set(response["target"]) == {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
        # the wrapper everything else calls returns the table alone
        assert pipe.auxiliary_observations(config, _retained()).equals(table)


def test_veith_both_representation_emits_potency_and_response_without_losing_potency(monkeypatch):
    """qhts_representation='both' routes Veith potency to the luminescent-AC50 assay and the
    max-response to the response head in one run, and a deposit that cannot supply the response
    (the old CYP1A2 schema) drops only the response, not that enzyme's potency rows."""
    import cypshift.pipeline as pipe
    from cypshift.data.assembly import RESPONSE_ASSAY

    monkeypatch.setattr("cypshift.data.sequences.sequence_for", lambda name, *a, **k: "M")
    monkeypatch.setattr("cypshift.data.auxiliary.site_distances", lambda scored, cands: None)
    monkeypatch.setattr("cypshift.data.auxiliary.select_auxiliaries", lambda *a, **k: [])

    def fake_veith(enzyme, smiles, *a, representation="pic50", **k):
        if representation == "max_response":
            if enzyme == "CYP1A2":  # old deposit: no Max_Response column
                raise KeyError("deposit has no 'Max_Response' column")
            return pd.DataFrame({"SMILES": ["CCO"], "value": [0.4], "target": enzyme})
        return pd.DataFrame({"SMILES": ["CCO"], "value": [5.0], "target": enzyme})

    monkeypatch.setattr("cypshift.data.veith.load_auxiliary", fake_veith)
    config = ExperimentConfig()
    config.override("model.kind", "graph")
    config.sources.veith = True
    config.override("sources.qhts_representation", "both")
    config.auxiliary.max_distance = 0.0  # scored enzymes only
    rows = pipe.auxiliary_observations(config, _retained())
    assays = set(rows["assay"])
    assert "luminescent_ac50" in assays and RESPONSE_ASSAY in assays  # both quantities present
    # CYP1A2 keeps a potency row though its response load raised.
    potency = rows[rows["assay"] == "luminescent_ac50"]
    assert "CYP1A2" in set(potency["target"])
    assert "CYP1A2" not in set(rows[rows["assay"] == RESPONSE_ASSAY]["target"])


def test_tox21_takes_the_qhts_representation_and_gives_cyp1a2_a_response(monkeypatch):
    """Tox21 is a qHTS source: max_response puts its rows on the response assay, both keeps the
    potency rows as well, and Tox21 supplies the CYP1A2 response the old Veith deposit lacks."""
    import cypshift.pipeline as pipe
    from cypshift.data.assembly import RESPONSE_ASSAY

    monkeypatch.setattr("cypshift.data.sequences.sequence_for", lambda name, *a, **k: "M")
    monkeypatch.setattr("cypshift.data.auxiliary.site_distances", lambda scored, cands: None)
    monkeypatch.setattr("cypshift.data.auxiliary.select_auxiliaries", lambda *a, **k: [])

    def fake_veith(enzyme, smiles, *a, representation="pic50", **k):
        if representation == "max_response" and enzyme == "CYP1A2":
            raise KeyError("deposit has no 'Max_Response' column")
        return pd.DataFrame({"SMILES": ["CCO"], "value": [0.4], "target": enzyme})

    def fake_tox21(enzyme, smiles, *a, representation="pic50", **k):
        value = 0.7 if representation == "max_response" else 5.5
        return pd.DataFrame({"SMILES": ["CCN"], "value": [value], "target": enzyme})

    monkeypatch.setattr("cypshift.data.veith.load_auxiliary", fake_veith)
    monkeypatch.setattr("cypshift.data.tox21.load_auxiliary", fake_tox21)
    for representation in ("max_response", "both"):
        config = ExperimentConfig()
        config.override("model.kind", "graph")
        config.sources.veith = True
        config.sources.tox21 = True
        config.override("sources.qhts_representation", representation)
        config.auxiliary.max_distance = 0.0
        rows = pipe.auxiliary_observations(config, _retained())
        response = rows[rows["assay"] == RESPONSE_ASSAY]
        assert set(response["target"]) == {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
        assert set(response[response["target"] == "CYP1A2"]["SMILES"]) == {"CCN"}  # Tox21 only
        assert ("luminescent_ac50" in set(rows["assay"])) == (representation == "both")

    tox21_only = ExperimentConfig()
    tox21_only.override("model.kind", "neural")
    tox21_only.sources.tox21 = True
    tox21_only.override("sources.qhts_representation", "max_response")
    assert pipe.response_feature_route(tox21_only)


def test_auxiliary_is_absent_when_the_source_is_unreachable(monkeypatch):
    """A run must not fail for lack of an auxiliary source (no network, no cache).

    Both paths are made to fail: relative selection (sequence_for) and the loader that
    pulls the scored enzymes' own distance-0 rows.
    """
    import cypshift.pipeline as pipe

    def boom(name, *a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr("cypshift.data.sequences.sequence_for", boom)
    monkeypatch.setattr("cypshift.data.chembl.load_auxiliary", boom)
    config = ExperimentConfig()
    config.sources.chembl = True
    assert pipe.auxiliary_observations(config, _retained()).empty


def test_run_widens_the_target_vocabulary_for_auxiliary_enzymes(monkeypatch):
    """Predictions stay on the scored four even when training on more targets."""
    import cypshift.pipeline as pipe

    captured = {}
    real_build = pipe.build_model

    def spy(config, screen_model=None, targets=None):
        captured["targets"] = targets
        return real_build(config, screen_model, targets=targets)

    monkeypatch.setattr(pipe, "build_model", spy)
    monkeypatch.setattr(
        pipe,
        "auxiliary_observations",
        lambda config, retained: pd.DataFrame(
            {
                "Molecule_Name": ["CYP2C19:0"],
                "SMILES": [retained["SMILES"].iloc[0]],
                "target": ["CYP2C19"],
                "assay": ["direct_inhibition_drc"],
                "value": [5.0],
                "conf_low": [5.0],
                "conf_high": [5.0],
                "weight": [0.6],
                "is_censored": [False],
                "data_space": ["distant_aux"],
            }
        ),
    )
    config = ExperimentConfig()
    config.override("model.kind", "boosted")
    config.model.boosted.n_estimators = 4
    config.features.fingerprint_bits = 256
    scores, prediction = pipe.run(config)
    assert "CYP2C19" in captured["targets"]
    assert prediction.mean.shape[1] == 4  # still only the scored endpoints


def test_embedding_conditioning_is_neural_only():
    """A tree cannot embed; the graph already linear-projects its one-hot. Only the
    neural model offers a learned (target, assay) table."""
    from cypshift.pipeline import supported_conditioning

    assert "embedding" in supported_conditioning("neural")
    assert "embedding" not in supported_conditioning("boosted")
    assert "embedding" not in supported_conditioning("graph")


def test_embedding_conditioning_is_rejected_for_a_tree():
    config = ExperimentConfig()
    config.override("model.kind", "boosted")
    config.override("conditioning.strategy", "embedding")
    with pytest.raises(ValueError, match="not available"):
        build_model(config)


def test_neural_builds_an_embedding_table_when_asked():
    config = ExperimentConfig()
    config.override("model.kind", "neural")
    config.override("conditioning.strategy", "embedding")
    config.conditioning.embedding_dim = 8
    model = build_model(config)
    assert model.conditioning == "embedding"
    assert model.embedding_dim == 8


def test_include_scored_adds_rows_for_the_scored_enzymes(monkeypatch):
    """The direct-augmentation path: the source's data for the scored endpoints
    themselves, not only non-scored relatives."""
    import cypshift.pipeline as pipe

    monkeypatch.setattr(
        "cypshift.data.chembl.load_auxiliary",
        lambda enzyme, smiles, *a, **k: pd.DataFrame(
            {"SMILES": ["c1ccccc1"], "value": [6.0], "target": enzyme}
        ),
    )
    config = ExperimentConfig()
    config.sources.chembl = True
    config.auxiliary.max_distance = 0.0  # distance 0 only: the scored enzymes' own rows
    config.auxiliary.scored_weight = 0.5
    rows = pipe.auxiliary_observations(config, _retained())
    assert set(rows["target"]) == {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
    assert np.allclose(rows["weight"], 0.5)


def test_auxiliary_dedups_against_the_whole_challenge_not_just_retained(monkeypatch):
    """A source row matching a held-out or blinded-test compound must never enter
    training. The loader is handed the full challenge SMILES, so assert that."""
    import cypshift.pipeline as pipe

    captured = {}

    def fake_loader(enzyme, smiles, *a, **k):
        captured["n_dedup_smiles"] = len(smiles)
        return pd.DataFrame({"SMILES": [], "value": [], "target": []})

    monkeypatch.setattr("cypshift.data.chembl.load_auxiliary", fake_loader)
    config = ExperimentConfig()
    config.sources.chembl = True
    config.auxiliary.max_distance = 0.0
    pipe.auxiliary_observations(config, _retained(n=5))
    # the dedup set is the full challenge (train + test), far larger than the 5 retained
    assert captured["n_dedup_smiles"] > 100


def test_macro_balance_equalises_scored_endpoint_weight():
    """Each scored endpoint should carry equal total weight; relatives stay untouched."""
    from cypshift.pipeline import balance_macro_weights

    obs = pd.DataFrame(
        {
            "target": ["CYP1A2"] * 10 + ["CYP2C9"] * 2 + ["CYP3A4"] * 6 + ["CYP2C19"] * 5,
            "weight": [1.0] * 23,
        }
    )
    out = balance_macro_weights(obs)
    scored = out[out["target"].isin(["CYP1A2", "CYP2C9", "CYP3A4"])]
    totals = scored.groupby("target")["weight"].sum()
    assert totals.max() - totals.min() < 1e-9
    assert (out.loc[out["target"] == "CYP2C19", "weight"] == 1.0).all()


def test_pool_same_assay_never_fabricates_a_measurement():
    """A real measurement supersedes a censored bound; the two are never averaged (F2)."""
    from cypshift.pipeline import pool_same_assay

    mixed = pd.DataFrame(
        {
            "SMILES": ["CCO", "OCC"],  # same compound, two spellings
            "value": [4.0, 8.0],
            "conf_low": [4.0, 8.0],
            "conf_high": [4.0, 8.0],
            "is_censored": [True, False],
        }
    )
    out = pool_same_assay(mixed)
    assert len(out) == 1  # canonical grouping merges CCO/OCC
    assert out["value"].iloc[0] == 8.0 and not out["is_censored"].iloc[0]

    censored = pd.DataFrame(
        {
            "SMILES": ["CCC", "CCC"],
            "value": [4.2, 4.0],
            "conf_low": [4.2, 4.0],
            "conf_high": [4.2, 4.0],
            "is_censored": [True, True],
        }
    )
    pooled = pool_same_assay(censored)
    assert pooled["value"].iloc[0] == 4.0 and pooled["is_censored"].iloc[0]


def test_save_run_bundle_writes_oof_predictions_and_metadata(tmp_path):
    """The bundle must carry per-compound OOF predictions beside truth, plus config/scores."""
    import json as _json

    from cypshift.config import REGRESSION_ENDPOINTS
    from cypshift.experiment import ExperimentConfig
    from cypshift.pipeline import save_run_bundle

    names = ["A", "B", "C"]
    evaluation = pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        evaluation[endpoint] = [5.0, 6.0, 4.0]
        evaluation[f"{endpoint}_conf_low"] = evaluation[endpoint] - 0.2
        evaluation[f"{endpoint}_conf_high"] = evaluation[endpoint] + 0.2
    submission = pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        submission[endpoint] = [4.8, 6.1, 4.2]
    scores = pd.DataFrame({"ST-RAE_mean": [0.6]}, index=["MA"])
    scores.index.name = "endpoint"

    bundle = tmp_path / "bundle"
    save_run_bundle(bundle, ExperimentConfig.load(), evaluation, submission, scores)
    oof = pd.read_parquet(bundle / "out_of_fold.parquet")
    assert len(oof) == 3
    assert f"{REGRESSION_ENDPOINTS[0]}_pred" in oof.columns  # prediction beside truth
    meta = _json.loads((bundle / "run.json").read_text())
    assert "config" in meta and "scores" in meta


def test_save_run_bundle_writes_the_pooled_predictions_beside_the_fine_tuned_ones(tmp_path):
    from cypshift.config import REGRESSION_ENDPOINTS
    from cypshift.experiment import ExperimentConfig
    from cypshift.pipeline import save_run_bundle

    names = ["A", "B"]
    evaluation = pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        evaluation[endpoint] = [5.0, 6.0]
        evaluation[f"{endpoint}_conf_low"] = evaluation[endpoint] - 0.2
        evaluation[f"{endpoint}_conf_high"] = evaluation[endpoint] + 0.2
    tuned, pooled = pd.DataFrame({"Molecule_Name": names}), pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        tuned[endpoint], pooled[endpoint] = [4.9, 6.1], [4.5, 6.5]
    scores = pd.DataFrame({"ST-RAE_mean": [0.6]}, index=["MA"])
    scores.index.name = "endpoint"
    save_run_bundle(tmp_path / "a", ExperimentConfig.load(), evaluation, tuned, scores, pooled)
    save_run_bundle(tmp_path / "b", ExperimentConfig.load(), evaluation, tuned, scores)
    column = f"{REGRESSION_ENDPOINTS[0]}_pred"
    assert pd.read_parquet(tmp_path / "a" / "out_of_fold_pooled.parquet")[column].tolist() == [
        4.5,
        6.5,
    ]
    assert pd.read_parquet(tmp_path / "a" / "out_of_fold.parquet")[column].tolist() == [4.9, 6.1]
    assert not (tmp_path / "b" / "out_of_fold_pooled.parquet").exists()


def test_r2_decomposition_joins_predictions_by_molecule_not_position():
    """F01 guard: the pooled-CV submission is in fold order, so the decomposition must join on
    Molecule_Name. Perfect predictions supplied in reverse row order still score R2 = 1."""
    from cypshift.config import REGRESSION_ENDPOINTS
    from cypshift.pipeline import _log_r2_decomposition

    names = ["a", "b", "c", "d"]
    values = [4.0, 5.0, 6.0, 7.0]
    evaluation = pd.DataFrame({"Molecule_Name": names})
    submission = pd.DataFrame({"Molecule_Name": names[::-1]})
    for endpoint in REGRESSION_ENDPOINTS:
        evaluation[endpoint] = values
        evaluation[f"{endpoint}_conf_low"] = values
        evaluation[f"{endpoint}_conf_high"] = values
        submission[endpoint] = values[::-1]  # perfect once joined by ID, catastrophic positionally
    table = _log_r2_decomposition(submission, evaluation)
    for endpoint in REGRESSION_ENDPOINTS:
        assert table.loc[endpoint.split("_")[0], "r2"] == pytest.approx(1.0)


def test_auxiliary_heads_are_columns_per_endpoint_and_assay_and_a_model_without_them_has_none():
    from cypshift.config import REGRESSION_ENDPOINTS
    from cypshift.pipeline import auxiliary_heads

    class WithHeads:
        def predict_auxiliary(self, smiles):
            return {"luminescent_ac50": np.arange(2 * 4, dtype=float).reshape(2, 4)}

    evaluation = pd.DataFrame({"Molecule_Name": ["A", "B"], "SMILES": ["CCO", "CCN"]})
    frame = auxiliary_heads(WithHeads(), evaluation)
    assert list(frame.columns) == [
        "Molecule_Name",
        *(f"{endpoint}__luminescent_ac50" for endpoint in REGRESSION_ENDPOINTS),
    ]
    assert frame[f"{REGRESSION_ENDPOINTS[1]}__luminescent_ac50"].tolist() == [1.0, 5.0]
    assert auxiliary_heads(object(), evaluation) is None


def test_save_run_bundle_writes_the_auxiliary_heads_beside_the_predictions_when_given(tmp_path):
    from cypshift.config import REGRESSION_ENDPOINTS
    from cypshift.experiment import ExperimentConfig
    from cypshift.pipeline import save_run_bundle

    names = ["A", "B"]
    evaluation = pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        evaluation[endpoint] = [5.0, 6.0]
        evaluation[f"{endpoint}_conf_low"] = evaluation[endpoint] - 0.2
        evaluation[f"{endpoint}_conf_high"] = evaluation[endpoint] + 0.2
    submission = pd.DataFrame({"Molecule_Name": names})
    for endpoint in REGRESSION_ENDPOINTS:
        submission[endpoint] = [4.9, 6.1]
    auxiliary = pd.DataFrame({"Molecule_Name": names, "x__y": [1.0, 2.0]})
    scores = pd.DataFrame({"ST-RAE_mean": [0.6]}, index=["MA"])
    scores.index.name = "endpoint"
    config = ExperimentConfig.load()
    save_run_bundle(tmp_path / "a", config, evaluation, submission, scores, None, auxiliary)
    save_run_bundle(tmp_path / "b", config, evaluation, submission, scores)
    assert pd.read_parquet(tmp_path / "a" / "out_of_fold_auxiliary.parquet")["x__y"].tolist() == [
        1.0,
        2.0,
    ]
    assert not (tmp_path / "b" / "out_of_fold_auxiliary.parquet").exists()
    # the predictions file is the same with or without the auxiliary heads
    assert pd.read_parquet(tmp_path / "a" / "out_of_fold.parquet").equals(
        pd.read_parquet(tmp_path / "b" / "out_of_fold.parquet")
    )
