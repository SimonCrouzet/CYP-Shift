"""Tests for the conserved P450 signature scan.

The scan is calibrated on the nine reviewed sequences, so these tests use short
synthetic sequences with the motifs placed at known offsets. That keeps them fast and
offline, and it tests the spacing logic rather than re-asserting UniProt's contents.
"""

import pytest

from cypshift.data.motifs import (
    HEME_LOOP,
    MOTIFS,
    Motif,
    describe,
    is_cytochrome_p450,
    p450_score,
    scan,
)

HEME = "FSAGKRICVG"
I_HELIX = "AGTETT"
EXXR = "EVQR"
FILLER = "LKLKLKLKLK"


def build(*, i_helix: bool = True, exxr: bool = True, length: int = 500) -> str:
    """A synthetic sequence with motifs at the offsets the scan expects."""
    residues = list((FILLER * 60)[:length])
    anchor = 400
    residues[anchor : anchor + len(HEME)] = HEME
    if i_helix:
        start = anchor - 131
        residues[start : start + len(I_HELIX)] = I_HELIX
    if exxr:
        start = anchor - 75
        residues[start : start + len(EXXR)] = EXXR
    return "".join(residues)


def test_a_complete_signature_scores_one():
    assert p450_score(build()) == pytest.approx(1.0)


def test_the_heme_loop_alone_does_not_pass():
    """A single ten-residue pattern can occur by chance in a long sequence."""
    lone = build(i_helix=False, exxr=False)
    assert p450_score(lone) == pytest.approx(0.5)
    assert not is_cytochrome_p450(lone)


def test_one_correctly_spaced_signature_is_enough():
    assert is_cytochrome_p450(build(exxr=False))
    assert is_cytochrome_p450(build(i_helix=False))


def test_no_heme_loop_scores_zero():
    assert p450_score(FILLER * 50) == 0.0


def test_a_motif_at_the_wrong_distance_is_not_counted():
    """Spacing is what discriminates: E[A-Z][A-Z]R is common, helix K is not."""
    residues = list(build(exxr=False))
    residues[10 : 10 + len(EXXR)] = EXXR  # far upstream of the expected window
    assert scan("".join(residues)).positions["exxr"] is None


def test_the_last_heme_loop_anchors_the_scan():
    """An accidental early match must not displace the real anchor."""
    sequence = HEME + build()
    assert scan(sequence).heme_loop == len(HEME) + 400


def test_motifs_are_found_at_the_expected_offsets():
    found = scan(build())
    assert found.heme_loop - found.positions["i_helix"] == 131
    assert found.heme_loop - found.positions["exxr"] == 75


@pytest.mark.parametrize("motif", MOTIFS, ids=lambda m: m.name)
def test_every_window_is_a_positive_range(motif: Motif):
    assert 0 < motif.low < motif.high


def test_heme_loop_matches_the_real_cysteine_context():
    """The pattern must accept the actual loop from a known P450."""
    import re

    assert re.search(HEME_LOOP, "PFSAGRRACLGEP")  # CYP2D6
    assert re.search(HEME_LOOP, "PFGSGPRNCIGMR")  # CYP3A4


def test_describe_reports_a_missing_heme_loop():
    assert "not a P450" in describe(FILLER * 50)


def test_describe_names_every_motif_when_present():
    text = describe(build())
    assert all(motif.name in text for motif in MOTIFS)
