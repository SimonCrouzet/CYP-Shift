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

from cypshift.models.graph import (
    CHEMELEON_URL,
    LOWER,
    UPPER,
    VALUE,
    GraphNetwork,
    RankingSoftThresholdLoss,
    SoftThresholdLoss,
    _TargetAdaptedRegressionFFN,
    chemeleon_message_passing,
)

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
