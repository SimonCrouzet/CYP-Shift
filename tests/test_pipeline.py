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


def test_screen_off_uses_only_the_curves():
    config = ExperimentConfig()
    config.sources.screen = False
    observations, surrogate = assemble_training_data(config, _retained())
    assert set(observations["assay"]) == {"direct_inhibition_drc"}
    assert surrogate is None


def test_an_unknown_strategy_says_so_rather_than_falling_back():
    """Declaring an option in the config must not silently do something else."""
    config = ExperimentConfig()
    config.multifidelity.screen_strategy = "distillation"
    with pytest.raises(ValueError, match="unknown screen strategy"):
        assemble_training_data(config, _retained())


@pytest.mark.parametrize("strategy", ["predicted_label", "embedding"])
@pytest.mark.parametrize("kind", ["graph", "curve"])
def test_a_surrogate_strategy_is_refused_where_it_makes_no_sense(kind, strategy):
    """A surrogate predicts what the screen would have said, for a model that cannot
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
def test_scalar_models_keep_the_surrogate_strategies(kind):
    """They predict a pIC50 and so cannot consume a response measurement; the surrogate
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


def test_an_empty_preincubated_intersection_is_treated_as_absent():
    """Synthetic compounds share no name with the preincubated file. An empty frame
    left nothing to concatenate; absent is what an empty intersection means."""
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_arm = True
    observations, _ = assemble_training_data(config, _retained())
    assert "tdi_drc" not in set(observations["assay"])


def test_the_preincubated_arm_reaches_the_training_table():
    """Omitting it reduced the classification track to a constant "no": the curve
    model's second arm trained on nothing, so predict_is_time_dependent compared an
    untrained arm against a trained one and scored MCC 0.0 -- identical to answering
    "no" to every compound. build_observations accepted a tdi argument that the
    pipeline never passed."""
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_arm = True
    observations, _ = assemble_training_data(config, _retained_with_tdi())
    assert "tdi_drc" in set(observations["assay"])


def test_the_arm_is_off_by_default_to_protect_the_regression_track():
    """It costs about 0.04 macro ST-RAE at every weight tried, more than every gain
    measured so far combined. The classification track turns it on instead."""
    assert ExperimentConfig().sources.tdi_arm is False


def test_the_preincubated_arm_can_be_switched_off():
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_arm = False
    observations, _ = assemble_training_data(config, _retained_with_tdi())
    assert "tdi_drc" not in set(observations["assay"])


def test_the_preincubated_arm_is_restricted_to_retained_compounds():
    """A holdout compound's preincubated measurement must not enter training."""
    retained = _retained_with_tdi()
    config = ExperimentConfig()
    config.override("model.kind", "curve")
    config.sources.tdi_arm = True
    observations, _ = assemble_training_data(config, retained)
    arm = observations[observations["assay"] == "tdi_drc"]
    assert set(arm["Molecule_Name"]) <= set(retained["Molecule_Name"])


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

    def spy(config, surrogate=None, targets=None):
        captured["targets"] = targets
        return real_build(config, surrogate, targets=targets)

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
                "tier": ["distant_aux"],
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
