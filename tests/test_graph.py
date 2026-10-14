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


@pytest.mark.skipif(MAMMAL_CACHE.exists(), reason="MAMMAL cache present; absent-path not testable")
def test_mammal_pretraining_needs_its_cache():
    """Without the MAMMAL embedding cache, construction fails loudly, never silently."""
    with pytest.raises(FileNotFoundError):
        GraphNetwork(pretraining="mammal")


def test_descriptor_blocks_widen_the_conditioning():
    """Computed descriptors the graph cannot learn are appended to its conditioning slot."""
    base = GraphNetwork(targets=("CYP2D6",))._n_conditioning
    wide = GraphNetwork(
        targets=("CYP2D6",), physchem=True, mechanism=True, ionization_geometry=True
    )._n_conditioning
    assert wide > base
