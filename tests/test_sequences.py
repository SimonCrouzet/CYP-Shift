"""Tests for sequence retrieval and FASTA parsing."""

from pathlib import Path

import pytest

from cypshift.data.sequences import ACCESSIONS, parse_fasta, read_fasta, sequence_for


def test_every_scored_enzyme_has_a_pinned_accession():
    assert {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"} <= set(ACCESSIONS)


def test_accessions_are_unique():
    """Two enzymes sharing an accession would silently share a descriptor."""
    assert len(set(ACCESSIONS.values())) == len(ACCESSIONS)


def test_parse_joins_wrapped_lines_and_drops_the_header():
    assert parse_fasta(">sp|P00000|TEST\nMKV\nLLA\n") == "MKVLLA"


def test_parse_rejects_a_header_with_no_sequence():
    with pytest.raises(ValueError, match="no sequence"):
        parse_fasta(">sp|P00000|TEST\n")


def test_parse_strips_alignment_gaps():
    assert parse_fasta(">x\nMK-V\n") == "MKV"


def test_unpinned_enzyme_is_refused():
    """Searching UniProt by name is not stable across releases."""
    unpinned = "CYP51A1"
    assert unpinned not in ACCESSIONS, "pick a name that is genuinely not pinned"
    with pytest.raises(KeyError, match="pinned"):
        sequence_for(unpinned)


def test_read_fasta_keys_on_the_first_header_field(tmp_path: Path):
    path = tmp_path / "two.fasta"
    path.write_text(">first description here\nMKV\n>second\nLLA\n")
    assert read_fasta(path) == {"first": "MKV", "second": "LLA"}


def test_read_fasta_handles_an_empty_file(tmp_path: Path):
    path = tmp_path / "empty.fasta"
    path.write_text("")
    assert read_fasta(path) == {}
