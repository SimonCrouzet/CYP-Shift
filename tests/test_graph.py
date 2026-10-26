"""Graph (D-MPNN) ensemble and pretrained-encoder wiring.

The actual fits are exercised by manual smoke runs, not here -- chemprop training is slow
and the CheMeleon download is 35 MB -- so these cover the fast, deterministic parts: the
encoder loads with the right shape, and an absent cache fails loudly rather than silently.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cypshift.models.base import ResolvedStage
from cypshift.models.graph import (
    CHEMELEON_URL,
    LOWER,
    UPPER,
    VALUE,
    GraphNetwork,
    RankingSoftThresholdLoss,
    SoftThresholdLoss,
    _FoldMonitor,
    _TargetAdaptedRegressionFFN,
    chemeleon_message_passing,
)
from cypshift.models.monitor import PrunedError

CHEMELEON_CACHE = Path("data/external/chemeleon/chemeleon_mp.pt")
MAMMAL_CACHE = Path("data/external/embeddings/mammal.parquet")


def test_chemeleon_url_points_at_zenodo():
    assert CHEMELEON_URL.startswith("https://zenodo.org/")


@pytest.mark.skipif(not CHEMELEON_CACHE.exists(), reason="CheMeleon encoder not downloaded")
def test_chemeleon_encoder_loads_with_its_published_width():
    encoder = chemeleon_message_passing(CHEMELEON_CACHE)
    assert encoder.output_dim == 2048  # CheMeleon's published hidden width


def _ffn_hidden_width(model) -> int:
    """The readout FFN's first hidden width (first Linear's out_features)."""
    linears = [m for m in model.predictor.ffn.modules() if m.__class__.__name__ == "Linear"]
    return linears[0].out_features


def test_ffn_hidden_sets_the_readout_width_independently_of_the_encoder():
    """model.graph.ffn_hidden widens the readout FFN without touching the message-passing width;
    0 keeps chemprop's default (300)."""
    assert _ffn_hidden_width(GraphNetwork(ffn_hidden=777)._build()) == 777
    assert _ffn_hidden_width(GraphNetwork()._build()) == 300  # default, unset


def test_graph_learning_rate_sets_the_schedule_and_the_default_keeps_chemprops():
    """``model.graph.learning_rate`` is the peak of the warmup/decay schedule, with the initial and
    final rates a tenth of it; unset it leaves chemprop's own 1e-4 / 1e-3 / 1e-4 and 2 warmup
    epochs, so existing configs are unchanged."""
    default = GraphNetwork()._build()
    assert (default.init_lr, default.max_lr, default.final_lr) == (1e-4, 1e-3, 1e-4)
    assert default.warmup_epochs == 2

    tuned = GraphNetwork(learning_rate=3e-4, warmup_epochs=4)._build()
    assert tuned.max_lr == pytest.approx(3e-4)
    assert tuned.init_lr == pytest.approx(3e-5) and tuned.final_lr == pytest.approx(3e-5)
    assert tuned.warmup_epochs == 4


def test_the_graph_learning_rate_reaches_the_model_from_the_config():
    from cypshift.experiment import ExperimentConfig
    from cypshift.pipeline import build_model

    config = ExperimentConfig()
    config.override("model.kind", "graph")
    config.override("model.graph.learning_rate", 1e-4)
    assert build_model(config).learning_rate == pytest.approx(1e-4)
    config.override("model.graph.learning_rate", -1.0)
    with pytest.raises(ValueError, match="non-negative"):
        config.validate()


@pytest.mark.skipif(not CHEMELEON_CACHE.exists(), reason="CheMeleon encoder not downloaded")
def test_freeze_toggle_controls_encoder_gradients():
    """pretraining_finetune=False freezes the CheMeleon encoder; the default trains it."""
    frozen = GraphNetwork(pretraining="chemeleon", pretraining_finetune=False)._build()
    assert all(not p.requires_grad for p in frozen.message_passing.parameters())
    trained = GraphNetwork(pretraining="chemeleon", pretraining_finetune=True)._build()
    assert any(p.requires_grad for p in trained.message_passing.parameters())


@pytest.mark.skipif(MAMMAL_CACHE.exists(), reason="MAMMAL cache present; absent-path not testable")
def test_mammal_pretraining_needs_its_cache():
    """Without the MAMMAL embedding cache, construction fails loudly, never silently."""
    with pytest.raises(FileNotFoundError):
        GraphNetwork(pretraining="mammal")


def test_use_structure_widens_the_conditioning_by_the_cofold_block():
    """use_structure appends the Boltz-2 cofold block to the graph's conditioning vector."""
    from cypshift.structure.cofold import STRUCTURE_FEATURE_NAMES

    base = GraphNetwork(targets=("CYP2D6",))._n_conditioning
    wide = GraphNetwork(targets=("CYP2D6",), use_structure=True)._n_conditioning
    assert wide == base + len(STRUCTURE_FEATURE_NAMES)


def test_source_heads_widen_the_conditioning_by_the_source_vocabulary():
    """source_heads appends a one-hot over SOURCES to the graph's conditioning vector."""
    from cypshift.data.assembly import SOURCES

    base = GraphNetwork(targets=("CYP2D6",))._n_conditioning
    wide = GraphNetwork(targets=("CYP2D6",), source_heads=True)._n_conditioning
    assert wide == base + len(SOURCES)


def test_separated_trunk_withholds_other_scored_isoforms_but_keeps_aux():
    """Each sub-model trains on its own isoform plus the auxiliary enzymes, never the other
    scored isoforms, so the trunks do not share a representation across scored targets."""
    from cypshift.models.graph import SeparatedTrunkGraph

    model = SeparatedTrunkGraph(
        endpoints=("CYP2D6_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition"),
        targets=("CYP2D6", "CYP3A4", "CYP2C19"),  # CYP2C19 is auxiliary, not scored
    )
    assert model._auxiliary == ("CYP2C19",)


def test_build_model_selects_separated_trunk_when_configured():
    from cypshift.experiment import ExperimentConfig
    from cypshift.models.graph import GraphNetwork, SeparatedTrunkGraph
    from cypshift.pipeline import build_model

    config = ExperimentConfig.load()
    config.override("model.kind", "graph")
    assert isinstance(build_model(config), GraphNetwork)
    config.override("model.graph.separate_trunks", True)
    assert isinstance(build_model(config), SeparatedTrunkGraph)


def test_descriptor_blocks_widen_the_conditioning():
    """Computed descriptors the graph cannot learn are appended to its conditioning slot."""
    base = GraphNetwork(targets=("CYP2D6",))._n_conditioning
    wide = GraphNetwork(
        targets=("CYP2D6",), physchem=True, mechanism=True, ionization_geometry=True
    )._n_conditioning
    assert wide > base


def test_a_response_assay_grows_the_label_matrix_and_assay_vocabulary():
    """A response head adds a fourth label column and joins the assay vocabulary; off by default."""
    plain = GraphNetwork()
    assert plain._n_label == 3
    for assay in ("luminescent_max_response", "direct_inhibition_screen_response"):
        head = GraphNetwork(response_assays=(assay,))
        assert head._n_label == 4
        assert assay in head.assays


def test_several_response_assays_each_get_their_own_label_column_and_scale():
    """Tier C step 1: the response heads coexist -- one label column per assay from RESPONSE on,
    each on its own 0-1 scale, and both assays join the vocabulary and the loss weights."""
    assays = ("luminescent_max_response", "direct_inhibition_screen_response")
    net = GraphNetwork(response_assays=assays)
    assert net._n_label == 3 + len(assays)
    assert all(a in net.assays for a in assays)
    assert net._resp_centre.shape == (len(assays),) and net._resp_spread.shape == (len(assays),)
    # The loss weights are 1 on the value task, 0 on the two carriers, 1 on each response task.
    weights = net._build().predictor.criterion.task_weights.squeeze()
    assert weights.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_per_target_heads_builds_one_value_head_per_target_on_a_shared_encoder():
    """Tier C step 4: per_target_heads gives every target its own value-task output head on the
    one shared message-passing encoder."""
    from cypshift.models.graph import _PerTargetHeadsRegressionFFN

    net = GraphNetwork(per_target_heads=True)
    mpnn = net._build()
    assert isinstance(mpnn.predictor, _PerTargetHeadsRegressionFFN)
    assert len(mpnn.predictor.heads) == len(net.targets)  # one head per target


def test_ffn_hidden_widens_the_per_target_heads():
    """P2 fix: ffn_hidden sets the per-target potency head width (not a fixed 128), so a
    per_target_heads + ffn_hidden sweep actually varies the potency readout, and the head carries
    the model dropout rather than none."""
    net = GraphNetwork(per_target_heads=True, ffn_hidden=256, dropout=0.2)
    head = net._build().predictor.heads[0]
    assert head[0].out_features == 256  # first Linear widened to ffn_hidden
    assert isinstance(head[2], torch.nn.Dropout) and head[2].p == pytest.approx(0.2)
    default = GraphNetwork(per_target_heads=True)._build().predictor.heads[0]
    assert default[0].out_features == 128  # ffn_hidden=0 keeps the 128 default


def test_per_target_heads_selects_the_row_s_own_head_by_the_target_one_hot():
    """The value prediction is the active target's head output; a different one-hot selects a
    different head, while the carrier/response columns come from the shared ffn."""
    net = GraphNetwork(per_target_heads=True)
    predictor = net._build().predictor
    offset = net._build().message_passing.output_dim
    # Give the two heads distinct constant outputs so the selection is observable.
    with torch.no_grad():
        for index, head in enumerate(predictor.heads):
            torch.nn.init.zeros_(head[0].weight)  # first Linear: hidden <- input
            torch.nn.init.zeros_(head[0].bias)
            torch.nn.init.zeros_(head[-1].weight)  # output Linear (after ReLU + Dropout)
            torch.nn.init.constant_(head[-1].bias, float(index + 1))
    z = torch.zeros(2, predictor.input_dim)
    z[0, offset + 0] = 1.0  # row 0 -> target 0's head (bias 1)
    z[1, offset + 2] = 1.0  # row 1 -> target 2's head (bias 3)
    value = predictor(z)[:, VALUE]
    assert value[0].item() == pytest.approx(1.0)
    assert value[1].item() == pytest.approx(3.0)


def test_adapter_correction_is_gated_to_its_target_row():
    """The per-isoform adapter changes only the value task of its own target's rows."""
    predictor = _TargetAdaptedRegressionFFN(
        adapt_hidden=4, target_offset=2, target_index=1, n_tasks=3, input_dim=5, dropout=0.0
    )
    # The last layer initialises to zero, so make the adapter non-trivial to probe the gate.
    for parameter in predictor.adapter.parameters():
        torch.nn.init.constant_(parameter, 0.5)
    z = torch.zeros(2, 5)
    z[0, 2 + 1] = 1.0  # row 0: the adapted target's one-hot is on (gate = 1)
    out = predictor(z)
    base = predictor.ffn(z)  # output_transform is identity here
    assert torch.allclose(out[1], base[1])  # gate off -> row unchanged
    assert not torch.allclose(out[0, VALUE], base[0, VALUE])  # gate on -> value shifts
    other = [t for t in range(3) if t != VALUE]
    assert torch.allclose(out[0, other], base[0, other])  # only the value task moves


def _bounds_targets(value: float, lower: float, upper: float) -> torch.Tensor:
    row = torch.zeros(1, 3)
    row[0, VALUE], row[0, LOWER], row[0, UPPER] = value, lower, upper
    return row


def test_point_term_pulls_inside_a_wide_band_but_spares_censored_rows():
    """point_weight adds a midpoint pull on a reliable row, and nothing on a censored row."""
    weights = torch.tensor([[1.0, 0.0, 0.0]])
    reliable = _bounds_targets(0.0, -1.0, 1.0)  # midpoint 0, wide band
    pred = torch.tensor([[0.5, 0.0, 0.0]])  # inside the band: dead-zone loss is zero
    mask = torch.ones(1, 3, dtype=torch.bool)
    off = SoftThresholdLoss(point_weight=0.0, task_weights=weights)
    on = SoftThresholdLoss(point_weight=1.0, task_weights=weights)
    assert float(off(pred, reliable, mask)) == pytest.approx(0.0, abs=1e-6)
    assert float(on(pred, reliable, mask)) > 0.0
    # A censored (one-sided) row carries the -1e4 lower sentinel and gets no point pull.
    censored = _bounds_targets(0.0, -1e4, 1.0)
    assert float(on(pred, censored, mask)) == pytest.approx(0.0, abs=1e-6)


def test_ranking_term_penalises_a_wrongly_ordered_disjoint_pair():
    """A disjoint pair ranked the wrong way costs more than the dead-zone loss alone."""
    weights = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.zeros(2, 3)
    targets[0] = torch.tensor([0.05, 0.0, 0.1])  # truly the lower compound
    targets[1] = torch.tensor([1.05, 1.0, 1.1])  # truly the higher compound
    mask = torch.ones(2, 3, dtype=torch.bool)
    wrong = torch.zeros(2, 3)
    wrong[:, VALUE] = torch.tensor([1.05, 0.05])  # predictions swapped: wrong order
    base = SoftThresholdLoss(task_weights=weights)
    ranked = RankingSoftThresholdLoss(ranking_weight=1.0, ranking_margin=0.5, task_weights=weights)
    assert float(ranked(wrong, targets, mask)) > float(base(wrong, targets, mask))
    # Correctly ordered predictions inside their own bands incur no ranking penalty.
    right = torch.zeros(2, 3)
    right[:, VALUE] = torch.tensor([0.05, 1.05])
    assert float(ranked(right, targets, mask)) == pytest.approx(
        float(base(right, targets, mask)), abs=1e-6
    )


def test_ranking_term_ignores_zero_weight_rows():
    """F04 guard: a wrongly ordered pair of zero-weight rows creates no ranking penalty."""
    weights = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.zeros(2, 3)
    targets[0] = torch.tensor([0.05, 0.0, 0.1])
    targets[1] = torch.tensor([1.05, 1.0, 1.1])
    mask = torch.ones(2, 3, dtype=torch.bool)
    wrong = torch.zeros(2, 3)
    wrong[:, VALUE] = torch.tensor([1.05, 0.05])  # wrong order
    ranked = RankingSoftThresholdLoss(ranking_weight=1.0, ranking_margin=0.5, task_weights=weights)
    base = SoftThresholdLoss(task_weights=weights)
    zero = torch.zeros(2)
    # With both rows at weight zero the ranking term must vanish (no gradient from nothing).
    assert float(ranked(wrong, targets, mask, zero)) == pytest.approx(
        float(base(wrong, targets, mask, zero)), abs=1e-6
    )


def test_ranking_term_scales_with_a_uniform_down_weight():
    """A uniform source down-weight must shrink the ranking contribution in the same proportion
    as the interval loss, not cancel out of the normalisation. Rows at weight 0.1 give a tenth
    of the ranking penalty they give at weight 1.0."""
    weights = torch.tensor([[1.0, 0.0, 0.0]])
    targets = torch.zeros(2, 3)
    targets[0] = torch.tensor([0.05, 0.0, 0.1])
    targets[1] = torch.tensor([1.05, 1.0, 1.1])
    mask = torch.ones(2, 3, dtype=torch.bool)
    wrong = torch.zeros(2, 3)
    wrong[:, VALUE] = torch.tensor([1.05, 0.05])  # wrong order
    ranked = RankingSoftThresholdLoss(ranking_weight=1.0, ranking_margin=0.5, task_weights=weights)
    base = SoftThresholdLoss(task_weights=weights)
    full, tenth = torch.ones(2), torch.full((2,), 0.1)
    contrib_full = float(ranked(wrong, targets, mask, full)) - float(
        base(wrong, targets, mask, full)
    )
    contrib_tenth = float(ranked(wrong, targets, mask, tenth)) - float(
        base(wrong, targets, mask, tenth)
    )
    assert contrib_full > 0
    assert contrib_tenth == pytest.approx(0.1 * contrib_full, rel=1e-5)


def test_censored_point_target_is_masked_under_point_objectives_only():
    """F07 guard: a censored row keeps its value for the soft-threshold dead-zone but is NaN
    (masked) for a two-sided point objective, which has no value to train it toward."""
    frame = pd.DataFrame(
        {
            "Molecule_Name": ["c0"],
            "SMILES": ["CCO"],
            "target": ["CYP1A2"],
            "assay": ["direct_inhibition_drc"],
            "value": [4.0],
            "conf_low": [4.0],
            "conf_high": [4.0],
            "weight": [1.0],
            "is_censored": [True],
            "source": ["challenge"],
        }
    )
    soft = GraphNetwork(loss="soft_threshold")._datapoints(frame, labelled=True)
    point = GraphNetwork(loss="absolute")._datapoints(frame, labelled=True)
    assert np.isfinite(soft[0].y[VALUE])  # dead-zone keeps the one-sided bound
    assert np.isnan(point[0].y[VALUE])  # the point objective drops the censored row


def _table_predictor(**kwargs):
    """A measurement-table graph and its predictor, with every head zeroed to a distinct constant
    (value head ``p`` outputs ``p + 1``; response task ``k``'s head for target ``i`` outputs
    ``100 * (k + 1) + i``) so which head a row selects can be read off the output."""
    net = GraphNetwork(per_measurement_heads=True, **kwargs)
    mpnn = net._build()
    predictor = mpnn.predictor
    with torch.no_grad():
        for heads in (predictor.value_heads, *predictor.response_heads):
            heads.w1.zero_(), heads.b1.zero_(), heads.w2.zero_()
        predictor.value_heads.b2.copy_(torch.arange(predictor.value_heads.b2.numel()) + 1.0)
        for task, heads in enumerate(predictor.response_heads):
            heads.b2.copy_(torch.arange(heads.b2.numel()) + 100.0 * (task + 1))
    predictor.eval()
    return net, mpnn.message_passing.output_dim, predictor


def test_measurement_table_has_one_potency_head_per_target_assay_pair():
    """Independent output table: a potency head per (target, assay) pair, a response head per
    target for each response task, all on the one shared encoder."""
    from cypshift.models.graph import _MeasurementTableRegressionFFN

    net = GraphNetwork(per_measurement_heads=True, response_assays=("luminescent_max_response",))
    predictor = net._build().predictor
    assert isinstance(predictor, _MeasurementTableRegressionFFN)
    assert predictor.value_heads.w1.shape[0] == len(net.targets) * len(net.assays)
    assert len(predictor.response_heads) == 1
    assert predictor.response_heads[0].w1.shape[0] == len(net.targets)


def test_measurement_table_selects_the_rows_own_target_assay_head():
    """Read through the real conditioning encoder: a (target, assay) pair selects its own head,
    and the same target under another assay selects a different one -- the independence that
    per_target_heads (one potency head per target, shared across assays) does not have."""
    from cypshift.data.assembly import encode_conditions

    net, offset, predictor = _table_predictor(response_assays=("luminescent_max_response",))
    frame = pd.DataFrame(
        {
            "target": ["CYP1A2", "CYP2D6", "CYP2D6", "CYP3A4"],
            "assay": [
                "direct_inhibition_drc",
                "direct_inhibition_drc",
                "tdi_drc",
                "luminescent_ac50",
            ],
        }
    )
    conditioning = torch.tensor(encode_conditions(frame, "factorised", net.targets, net.assays))
    out = predictor(torch.cat([torch.zeros(len(frame), offset), conditioning], dim=1))
    n_assays = len(net.assays)
    for row, (target, assay) in enumerate(zip(frame["target"], frame["assay"], strict=True)):
        expected = net.targets.index(target) * n_assays + net.assays.index(assay) + 1
        assert out[row, VALUE].item() == pytest.approx(expected)
        # the response column is selected by target alone
        assert out[row, 3].item() == pytest.approx(100 + net.targets.index(target))
    assert out[1, VALUE].item() != out[2, VALUE].item()  # CYP2D6 direct vs tdi: different heads


def test_measurement_table_head_width_follows_ffn_hidden_and_dropout():
    net = GraphNetwork(per_measurement_heads=True, ffn_hidden=96, dropout=0.25)
    predictor = net._build().predictor
    assert predictor.value_heads.w1.shape[2] == 96
    assert predictor.value_heads.dropout.p == pytest.approx(0.25)
    assert (
        GraphNetwork(per_measurement_heads=True)._build().predictor.value_heads.w1.shape[2] == 128
    )


def test_measurement_table_needs_the_factorised_one_hot_layout():
    with pytest.raises(ValueError, match="factorised"):
        GraphNetwork(per_measurement_heads=True, conditioning="metadata")


def test_measurement_table_trains_only_the_pairs_it_sees():
    """A real fit: heads for the pairs present in the data move; the head of a pair that never
    appears gets no gradient and is exactly unchanged -- the heads are independent, and the response
    head moves only for the target that carried response data."""
    import lightning.pytorch as pl

    response = "luminescent_max_response"
    rng = np.random.default_rng(0)
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCC", "c1ccncc1", "CCOCC", "CC(C)O"]
    rows = []
    for smi in smiles:
        for target, assay in (
            ("CYP2D6", "direct_inhibition_drc"),
            ("CYP3A4", "direct_inhibition_drc"),
            ("CYP3A4", "fluorogenic_pic50"),
            ("CYP2D6", response),
        ):
            value = float(rng.uniform(0, 1) if assay == response else rng.uniform(4, 7))
            spread = 0.0 if assay == response else 0.3
            rows.append(
                {
                    "SMILES": smi,
                    "target": target,
                    "assay": assay,
                    "source": "challenge",
                    "value": value,
                    "conf_low": value - spread,
                    "conf_high": value + spread,
                    "is_censored": False,
                    "weight": 1.0,
                }
            )
    settings = {
        "hidden": 32,
        "depth": 2,
        "ffn_hidden": 32,
        "response_assays": (response,),
        "per_measurement_heads": True,
    }
    net = GraphNetwork(epochs=3, batch_size=16, n_members=1, **settings)
    net.fit(pd.DataFrame(rows))
    trained = net._members[0].predictor
    pl.seed_everything(0, workers=True)  # the fit's member-0 seed, so this rebuild is its init
    initial = GraphNetwork(**settings)._build().predictor

    n_assays = len(net.assays)

    def pair(target: str, assay: str) -> int:
        return net.targets.index(target) * n_assays + net.assays.index(assay)

    def moved(index: int) -> float:
        delta = trained.value_heads.w2[index] - initial.value_heads.w2[index]
        return float(delta.detach().abs().max())

    for seen in (
        pair("CYP2D6", "direct_inhibition_drc"),
        pair("CYP3A4", "direct_inhibition_drc"),
        pair("CYP3A4", "fluorogenic_pic50"),
    ):
        assert moved(seen) > 1e-4
    for unseen in (
        pair("CYP1A2", "direct_inhibition_drc"),
        pair("CYP2D6", "tdi_drc"),
        pair("CYP2C9", "fluorogenic_pic50"),
    ):
        assert moved(unseen) == 0.0

    def response_moved(target: str) -> float:
        index = net.targets.index(target)
        delta = trained.response_heads[0].w2[index] - initial.response_heads[0].w2[index]
        return float(delta.detach().abs().max())

    assert response_moved("CYP2D6") > 1e-4 and response_moved("CYP1A2") == 0.0
    assert np.isfinite(net.predict(smiles[:3]).samples).all()


def _staged_rows(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O"]
    rows = []
    for i in range(n):
        space, assay = (
            ("challenge", "direct_inhibition_drc")
            if i % 3 == 0
            else ("near_aux", "fluorogenic_pic50")
        )
        value = float(rng.uniform(4, 7))
        rows.append(
            {
                "SMILES": smiles[i % len(smiles)],
                "target": "CYP3A4",
                "assay": assay,
                "source": "challenge",
                "data_space": space,
                "value": value,
                "conf_low": value - 0.3,
                "conf_high": value + 0.3,
                "is_censored": False,
                "weight": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _recorded_loaders(monkeypatch) -> list[int]:
    """Row counts of the datasets the graph builds a loader for, in order."""
    from cypshift.models import graph as graph_module

    sizes: list[int] = []
    original = graph_module.cpdata.build_dataloader

    def record(dataset, **kwargs):
        sizes.append(len(dataset))
        return original(dataset, **kwargs)

    monkeypatch.setattr(graph_module.cpdata, "build_dataloader", record)
    return sizes


def test_a_later_stage_trains_on_its_own_rows_only(monkeypatch):
    sizes = _recorded_loaders(monkeypatch)
    pooled = ResolvedStage(("distant_aux", "near_aux", "challenge"), None, 1, 0.0, False)
    direct = ResolvedStage(("challenge",), ("direct_inhibition_drc",), 1, 2e-4, False)
    net = GraphNetwork(
        hidden=16, depth=2, ffn_hidden=16, epochs=1, batch_size=8, stages=(pooled, direct)
    )
    net.fit(_staged_rows())
    assert sizes == [24, 8]
    assert np.isfinite(net.predict(["CCO", "CCN"]).mean).all()


def test_without_stages_one_pooled_fit_runs_over_every_row(monkeypatch):
    sizes = _recorded_loaders(monkeypatch)
    GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=1, batch_size=8).fit(_staged_rows())
    assert sizes == [24]


def test_a_stage_sets_its_schedule_and_whether_the_encoder_trains():
    net = GraphNetwork(hidden=16, depth=2, warmup_epochs=2)
    model = net._build()
    free = ResolvedStage(("challenge",), None, 5, 0.0, False)
    net._configure_stage(model, free, position=0)
    assert model.warmup_epochs == 2 and model.max_lr == 1e-3  # the model's own schedule
    frozen = ResolvedStage(("challenge",), None, 5, 2e-4, True)
    net._configure_stage(model, frozen, position=1)
    assert model.warmup_epochs == 0
    assert (model.init_lr, model.max_lr, model.final_lr) == pytest.approx((2e-5, 2e-4, 2e-5))
    assert not any(p.requires_grad for p in model.message_passing.parameters())
    assert all(p.requires_grad for p in model.predictor.parameters())
    net._configure_stage(model, free, position=2)
    assert all(p.requires_grad for p in model.message_passing.parameters())


def _monitored_rows(n: int = 40) -> pd.DataFrame:
    rows = _staged_rows(n)
    rows["data_space"] = "challenge"
    rows["assay"] = "direct_inhibition_drc"
    return rows


def test_a_monitor_is_evaluated_after_every_epoch_without_joining_the_training_rows(monkeypatch):
    sizes = _recorded_loaders(monkeypatch)
    net = GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=3, batch_size=8)
    net.fit(_monitored_rows(40), monitor=_monitored_rows(12))
    assert sizes == [40, 12]
    (curve,) = net.curves
    assert (curve["member"], curve["stage"], curve["epochs"]) == (1, 1, 3)
    assert len(curve["error"]) == 3 and np.isfinite(curve["error"]).all()


def test_without_a_monitor_nothing_is_evaluated_and_no_curve_is_kept(monkeypatch):
    sizes = _recorded_loaders(monkeypatch)
    net = GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=1, batch_size=8)
    net.fit(_monitored_rows(40))
    assert sizes == [40] and net.curves == []


def test_a_limit_below_the_halfway_error_prunes_the_fit_and_one_above_it_does_not():
    rows, monitor = _monitored_rows(40), _monitored_rows(12)
    with pytest.raises(PrunedError, match="fold error"):
        GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=4, batch_size=8).fit(
            rows, monitor=monitor, prune_above=0.0
        )
    net = GraphNetwork(hidden=16, depth=2, ffn_hidden=16, epochs=4, batch_size=8)
    net.fit(rows, monitor=monitor, prune_above=1e6)
    assert len(net.curves[0]["error"]) == 4


class _Trainer:
    should_stop = False


def _scripted_monitor(errors, **kwargs):
    """A monitor whose per-epoch errors are given instead of measured."""
    monitor = _FoldMonitor(None, 1.0, kwargs.get("patience", 0), kwargs.get("prune_at"), None)
    monitor.prune_above = kwargs.get("prune_above")
    scripted = iter(errors)
    monitor._measure = lambda module: next(scripted)
    return monitor


def _run_epochs(monitor, count):
    trainer = _Trainer()
    for _ in range(count):
        monitor.on_train_epoch_end(trainer, None)
        if trainer.should_stop:
            break
    return trainer.should_stop, len(monitor.error)


def test_a_stage_stops_after_patience_epochs_without_a_lower_error():
    stopped, epochs = _run_epochs(_scripted_monitor([5, 4, 4.5, 4.2, 4.1, 3], patience=3), 6)
    assert (stopped, epochs) == (True, 5)
    assert _run_epochs(_scripted_monitor([5, 4, 3, 2], patience=3), 4) == (False, 4)
    assert _run_epochs(_scripted_monitor([1, 2, 3, 4], patience=0), 4) == (False, 4)


def test_the_error_at_the_pruning_epoch_decides_pruning_only_there():
    monitor = _scripted_monitor([9, 1.4, 9], prune_at=2, prune_above=1.5)
    assert _run_epochs(monitor, 3) == (False, 3) and not monitor.pruned
    monitor = _scripted_monitor([1, 1.6, 9], prune_at=2, prune_above=1.5)
    assert _run_epochs(monitor, 3) == (True, 2) and monitor.pruned
