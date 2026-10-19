"""Graph (D-MPNN) ensemble and pretrained-encoder wiring.

The actual fits are exercised by manual smoke runs, not here -- chemprop training is slow
and the CheMeleon download is 35 MB -- so these cover the fast, deterministic parts: the
encoder loads with the right shape, and an absent cache fails loudly rather than silently.
"""

from pathlib import Path

import pytest

from cypshift.models.graph import CHEMELEON_URL, GraphNetwork, chemeleon_message_passing

CHEMELEON_CACHE = Path("data/external/chemeleon/chemeleon_mp.pt")
MAMMAL_CACHE = Path("data/external/embeddings/mammal.parquet")


def test_chemeleon_url_points_at_zenodo():
    assert CHEMELEON_URL.startswith("https://zenodo.org/")


@pytest.mark.skipif(not CHEMELEON_CACHE.exists(), reason="CheMeleon encoder not downloaded")
def test_chemeleon_encoder_loads_with_its_published_width():
    encoder = chemeleon_message_passing(CHEMELEON_CACHE)
    assert encoder.output_dim == 2048  # CheMeleon's published hidden width


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
