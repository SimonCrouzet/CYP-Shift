"""Tests for binding-site comparison and scanning."""

import pytest

from cypshift.data.site import (
    MAX_SHIFT,
    MIN_OVERLAP,
    align_region,
    charged_positions,
    find_motif,
    rank_by_site_similarity,
    region_identity,
    region_shifts,
    site_distance,
    site_identity,
    site_length,
)
from cypshift.data.srs import REGIONS, binding_site, region_sequences

HEME = "FSAGKRICVG"


def synthetic(filler: str = "LKLKLKLKLK", anchor: int = 400, length: int = 520) -> str:
    residues = list((filler * 60)[:length])
    residues[anchor : anchor + len(HEME)] = HEME
    return "".join(residues)


def test_site_length_is_the_sum_of_the_bands():
    assert site_length() == sum(r.high - r.low + 1 for r in REGIONS)


def test_binding_site_is_shorter_than_the_chain():
    """About 30% of it. A site spanning the whole protein would not be a site."""
    sequence = synthetic()
    assert 0 < len(binding_site(sequence)) < len(sequence)


def test_every_region_is_extracted_from_a_long_enough_sequence():
    assert set(region_sequences(synthetic())) == {r.name for r in REGIONS}


def test_a_sequence_without_a_heme_loop_is_refused():
    with pytest.raises(ValueError, match="no heme-binding loop"):
        region_sequences("LKLK" * 100)


def test_identical_sequences_are_fully_identical():
    sequence = synthetic()
    assert site_identity(sequence, sequence) == pytest.approx(1.0)
    assert site_distance(sequence, sequence) == pytest.approx(0.0)


def test_alignment_finds_no_shift_when_none_is_needed():
    assert align_region("ACDEFGHIKL", "ACDEFGHIKL") == (1.0, 0)


def test_alignment_recovers_a_shifted_region():
    """A single indel upstream slides a whole band; strict comparison would call two
    homologous regions unrelated."""
    score, shift = align_region("XACDEFGHIKL", "ACDEFGHIKLM")
    assert score == pytest.approx(1.0)
    assert shift == 1


def test_alignment_reports_nothing_when_overlap_is_too_short():
    assert align_region("ACD", "GHI") == (0.0, 0)


def test_a_short_band_cannot_be_slid_off_itself():
    """MAX_SHIFT alone does not bound the search: the narrowest band is only as wide
    as the shift window, so MIN_OVERLAP is what stops it sliding clean off."""
    narrowest = min(r.high - r.low + 1 for r in REGIONS)
    assert narrowest - MAX_SHIFT < MIN_OVERLAP
    assert align_region("A" * narrowest, "C" * narrowest) == (0.0, 0)


def test_a_tie_resolves_to_no_shift():
    """A repetitive band matches itself at several offsets; the largest is not an
    indel."""
    assert align_region("LKLKLKLKLKLKLKLK", "LKLKLKLKLKLKLKLK")[1] == 0


def test_region_identity_covers_every_extracted_region():
    sequence = synthetic()
    assert set(region_identity(sequence, sequence)) == {r.name for r in REGIONS}


def test_shifts_are_zero_for_a_sequence_against_itself():
    sequence = synthetic()
    assert set(region_shifts(sequence, sequence).values()) == {0}


def test_find_motif_reports_the_region_it_matched_in():
    sequence = synthetic(filler="LKLKLKLKLK")
    hits = find_motif(sequence, "LK")
    assert hits
    assert {hit.region for hit in hits} <= {r.name for r in REGIONS}


def test_find_motif_returns_nothing_for_an_absent_pattern():
    assert find_motif(synthetic(), "WWWWW") == ()


def test_find_motif_takes_an_arbitrary_pattern():
    """The point of the function: not limited to the fixed P450 signatures."""
    assert find_motif(synthetic(filler="LKLKLKLKLK"), "[LK]{4}")


def test_charged_positions_counts_basic_minus_acidic():
    basic = synthetic(filler="RRRRRRRRRR")
    assert all(value > 0 for value in charged_positions(basic).values())
    acidic = synthetic(filler="DDDDDDDDDD")
    assert all(value < 0 for value in charged_positions(acidic).values())


def test_ranking_puts_the_closest_candidate_first():
    target = synthetic(filler="LKLKLKLKLK")
    ranked = rank_by_site_similarity(
        target,
        {"same": target, "different": synthetic(filler="DWDWDWDWDW")},
    )
    assert ranked[0][0] == "same"


def test_ranking_skips_a_candidate_with_no_heme_loop():
    """Skipped with a log line, rather than ranked on a comparison that cannot be
    made."""
    target = synthetic()
    ranked = rank_by_site_similarity(target, {"ok": target, "broken": "LKLK" * 100})
    assert [name for name, _ in ranked] == ["ok"]


def test_ranking_respects_the_limit():
    target = synthetic()
    candidates = {f"c{i}": synthetic(filler="LKLKLKLKLK") for i in range(5)}
    assert len(rank_by_site_similarity(target, candidates, limit=2)) == 2
