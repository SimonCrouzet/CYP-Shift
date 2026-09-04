"""Tests for the per-region substrate recognition site encoding."""

import numpy as np
import pytest

from cypshift.data.pocket import PocketComposition
from cypshift.data.srs import (
    REGIONS,
    SRS_DESCRIPTOR_NAMES,
    assign_regions,
    region_for,
    srs_descriptor,
)

HEME = "FSAGKRICVG"
FILLER = "LKLKLKLKLK"


def sequence_with_heme_loop_at(anchor: int = 400, length: int = 520) -> str:
    residues = list((FILLER * 60)[:length])
    residues[anchor : anchor + len(HEME)] = HEME
    return "".join(residues)


def pocket_at(*residues: tuple[str, int]) -> PocketComposition:
    """A pocket whose residues sit at chosen sequence numbers."""
    return PocketComposition(
        "TEST",
        tuple(name for name, _ in residues),
        tuple(1.0 for _ in residues),
        tuple(number for _, number in residues),
    )


def test_descriptor_is_three_zscales_per_region():
    assert len(SRS_DESCRIPTOR_NAMES) == 3 * len(REGIONS)


def test_regions_do_not_overlap():
    """An overlap would place one residue in two regions and double-count it."""
    for first, second in zip(REGIONS, REGIONS[1:], strict=False):
        assert second.high < first.low


def test_regions_run_amino_to_carboxy():
    """Ordered N to C, which means decreasing offset from the heme loop."""
    offsets = [region.low for region in REGIONS]
    assert offsets == sorted(offsets, reverse=True)


def test_region_lookup_returns_none_outside_every_band():
    assert region_for(1000) is None


def test_srs6_lies_downstream_of_the_heme_loop():
    """The beta4 hairpin follows the heme loop in sequence, so its offset is negative."""
    assert REGIONS[-1].name == "SRS6"
    assert REGIONS[-1].high < 0


def test_a_residue_is_assigned_to_the_expected_region():
    sequence = sequence_with_heme_loop_at(400)
    # anchor is 1-based 401; a residue at 401 - 130 = 271 is 130 upstream, in SRS4
    grouped = assign_regions(pocket_at(("ASP", 271)), sequence)
    assert grouped == {"SRS4": ("ASP",)}


def test_residues_outside_every_band_are_dropped_not_misfiled():
    sequence = sequence_with_heme_loop_at(400)
    assert assign_regions(pocket_at(("ASP", 401)), sequence) == {}


def test_a_pocket_without_residue_numbers_is_refused():
    """Numbers are what place a residue in a region; counts alone cannot."""
    numberless = PocketComposition("TEST", ("ASP",), (1.0,))
    with pytest.raises(ValueError, match="no residue numbers"):
        assign_regions(numberless, sequence_with_heme_loop_at())


def test_a_sequence_without_a_heme_loop_is_refused():
    with pytest.raises(ValueError, match="no heme-binding loop"):
        assign_regions(pocket_at(("ASP", 271)), FILLER * 50)


def test_an_empty_region_stays_at_zero():
    """A region the ligand does not reach is a fact about the pocket, not a gap."""
    sequence = sequence_with_heme_loop_at(400)
    descriptor = srs_descriptor(pocket_at(("ASP", 271)), sequence)
    index = [r.name for r in REGIONS].index("SRS1") * 3
    assert np.allclose(descriptor[index : index + 3], 0.0)


def test_separating_regions_keeps_an_anchor_that_pooling_buries():
    """The whole argument for the module: one charged residue among many survives."""
    sequence = sequence_with_heme_loop_at(400)
    greasy = [("LEU", 401 - 130 + i) for i in range(6)]
    with_anchor = pocket_at(("ARG", 401 - 320), *greasy)
    without = pocket_at(("LEU", 401 - 320), *greasy)
    assert (
        np.linalg.norm(srs_descriptor(with_anchor, sequence) - srs_descriptor(without, sequence))
        > 3.0
    )


def test_descriptor_is_finite():
    sequence = sequence_with_heme_loop_at(400)
    assert np.isfinite(srs_descriptor(pocket_at(("ASP", 271)), sequence)).all()
