"""Comparing and scanning enzymes over the binding site rather than the whole chain.

Whole-sequence identity is the wrong measure for choosing what to pretrain on. Two
P450s share a scaffold that is invariant by construction -- the heme loop, the ExxR
salt bridge, the I-helix acid-alcohol pair -- and that shared scaffold inflates any
global similarity without saying anything about whether the two enzymes prefer the same
ligands. What predicts shared ligand preference is the part that varies, which is the
substrate recognition sites of :mod:`cypshift.data.srs`.

**No multiple alignment is needed, but a small shift search is.** Each band is a fixed
offset from the heme loop, and the heme loop is found in any P450 sequence, so every
enzyme yields a site string of the same length. Comparing those position for position
is not quite enough, though: a single insertion or deletion between the anchor and a
band slides that whole band by one, and a strict position-wise comparison then scores
two homologous regions as unrelated. Measured on CYP1A2 against CYP1A1, which are
same-family and closely related, strict comparison gave SRS4 an identity of 0.05 --
the two regions are ``VNLVNDIFGAGFDTVTTAISWS`` and ``IINIVLDLFGAGFDTVTTAISW``, plainly
homologous and off by one.

So each band is slid against its counterpart and scored at its best offset. That
recovers SRS4 to 0.81 and SRS1a to 0.90 for that pair, and changes nothing for
CYP2C9 against CYP2C19, which needs no shift anywhere. The offsets a pair needs are
consistent across its bands, which is what a single indel looks like. Only whole-band
shifts are searched, not gaps inside a band -- for a distant relative, read the
per-region scores rather than the summary number.

Two ways to compare are provided, and they answer different questions. Identity counts
exact matches and is the stricter one. Z-scale distance treats a conservative
substitution as a small change rather than a total mismatch, which is usually what you
want when the question is whether two pockets hold the same ligands.
"""

import logging
import re
from dataclasses import dataclass

import numpy as np

from cypshift.data.srs import REGIONS, region_sequences
from cypshift.data.zscales import ZSCALES_BY_LETTER, sequence_zscales

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotifHit:
    """Where a pattern matched inside the binding site."""

    region: str
    offset: int
    matched: str


#: How far a band may be slid against its counterpart. Ten covers the indels seen
#: between human CYP families; a larger window starts finding spurious alignments in
#: regions of twenty-odd residues.
MAX_SHIFT = 10

#: Below this many overlapping positions a match is not evidence of anything.
MIN_OVERLAP = 8


def _overlap(first: str, second: str, shift: int) -> tuple[str, str]:
    """The two subsequences that align when ``second`` is slid by ``shift``."""
    a = first[shift:] if shift > 0 else first
    b = second[-shift:] if shift < 0 else second
    width = min(len(a), len(b))
    return a[:width], b[:width]


def align_region(first: str, second: str, max_shift: int = MAX_SHIFT) -> tuple[float, int]:
    """Best identity between two bands, and the shift achieving it.

    Returns:
        ``(identity, shift)``, or ``(0.0, 0)`` if no offset leaves enough overlap.
    """
    best, argbest = 0.0, 0
    # Ordered by how far the shift moves the band, so a tie resolves to the smallest
    # displacement. Without this a repetitive band matches itself equally well at
    # several offsets and the largest one wins, which reads as an indel that is not
    # there. Shorter bands also allow less travel: MIN_OVERLAP is what stops a
    # ten-residue region being slid clean off itself.
    for shift in sorted(range(-max_shift, max_shift + 1), key=lambda value: (abs(value), value)):
        a, b = _overlap(first, second, shift)
        if len(a) < MIN_OVERLAP:
            continue
        score = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
        if score > best:
            best, argbest = score, shift
    return best, argbest


def region_identity(first: str, second: str) -> dict[str, float]:
    """Per-region identity, which says *where* two enzymes differ.

    A pair can share most of the site and still diverge at the one region that decides
    recognition, and a single number hides that. Each band is compared at its best
    offset, so a lone indel upstream does not read as a total mismatch.
    """
    left, right = region_sequences(first), region_sequences(second)
    scores: dict[str, float] = {}
    for region in REGIONS:
        a, b = left.get(region.name, ""), right.get(region.name, "")
        if a and b:
            scores[region.name] = align_region(a, b)[0]
    return scores


def region_shifts(first: str, second: str) -> dict[str, int]:
    """The offset each band needed. Diagnostic: a real indel gives consistent shifts,
    while scattered ones mean the bands are not lining up and the summary is unsafe."""
    left, right = region_sequences(first), region_sequences(second)
    return {
        region.name: align_region(left[region.name], right[region.name])[1]
        for region in REGIONS
        if region.name in left and region.name in right
    }


def site_identity(first: str, second: str) -> float:
    """Fraction of binding-site positions carrying the same residue.

    Averaged over bands, each compared at its best offset. Weighted by band length, so
    a sixteen-residue region does not count as much as a thirty-two-residue one.
    """
    left, right = region_sequences(first), region_sequences(second)
    matched = total = 0.0
    for region in REGIONS:
        a, b = left.get(region.name, ""), right.get(region.name, "")
        if not (a and b):
            continue
        score, shift = align_region(a, b)
        width = len(_overlap(a, b, shift)[0])
        matched += score * width
        total += width
    if not total:
        raise ValueError("the two sequences share no comparable binding-site region")
    return matched / total


def site_distance(first: str, second: str) -> float:
    """Mean Z-scale distance across binding-site positions.

    Unlike :func:`site_identity`, a conservative substitution counts as a small change.
    Leucine for isoleucine moves this barely; aspartate for arginine moves it a lot,
    which is the distinction that matters for whether two pockets bind alike.
    """
    left, right = region_sequences(first), region_sequences(second)
    distances: list[float] = []
    for region in REGIONS:
        a, b = left.get(region.name, ""), right.get(region.name, "")
        if not (a and b):
            continue
        shift = align_region(a, b)[1]
        x, y = _overlap(a, b, shift)
        table_a, table_b = sequence_zscales(x), sequence_zscales(y)
        if len(table_a) == len(table_b) and len(table_a):
            distances.extend(np.linalg.norm(table_a - table_b, axis=1))
    if not distances:
        raise ValueError("the two sequences share no comparable binding-site region")
    return float(np.mean(distances))


def find_motif(sequence: str, pattern: str) -> tuple[MotifHit, ...]:
    """Scan the binding site for any pattern, not just the conserved signatures.

    The signatures in :mod:`cypshift.data.motifs` are fixed because they answer a fixed
    question -- is this a P450. This takes an arbitrary regular expression, so a
    hypothesis about a recognition element can be checked directly: which enzymes carry
    an acidic residue in the I-helix groove, say, or a basic one in the B'-C loop.

    Args:
        sequence: A P450 sequence.
        pattern: Any regular expression over one-letter residues.

    Returns:
        One hit per match, carrying the region it fell in and its offset within that
        region. Matches outside the binding site are not reported.
    """
    hits: list[MotifHit] = []
    for name, subsequence in region_sequences(sequence).items():
        for match in re.finditer(f"(?=({pattern}))", subsequence):
            hits.append(MotifHit(name, match.start(), match.group(1)))
    return tuple(hits)


def charged_positions(sequence: str) -> dict[str, int]:
    """Net charge of each recognition site, as a quick read on recognition chemistry.

    The single most informative thing about a CYP pocket: an anion-pairing enzyme
    carries a basic residue where a cation-pairing one carries an acidic one.
    """
    counts: dict[str, int] = {}
    for name, subsequence in region_sequences(sequence).items():
        counts[name] = sum(r in "RK" for r in subsequence) - sum(r in "DE" for r in subsequence)
    return counts


def rank_by_site_similarity(
    target: str, candidates: dict[str, str], limit: int | None = None
) -> list[tuple[str, float]]:
    """Rank candidate enzymes by how close their binding site is to the target's.

    This is the selection criterion for pretraining: an auxiliary enzyme contributes
    data only if it recognises the same kind of molecule, and the binding site is where
    that is written. Candidates whose site cannot be compared -- no heme loop, or a
    different site length -- are skipped with a log line rather than ranked wrongly.

    Returns:
        ``(name, distance)`` pairs, closest first. Distance is the Z-scale measure of
        :func:`site_distance`, so smaller means more alike.
    """
    scored: list[tuple[str, float]] = []
    for name, sequence in candidates.items():
        try:
            scored.append((name, site_distance(target, sequence)))
        except ValueError as error:
            logger.info("skipping %s: %s", name, error)
    scored.sort(key=lambda row: row[1])
    return scored[:limit] if limit else scored


def site_length() -> int:
    """How many residues a binding site spans, fixed across enzymes by construction."""
    return sum(region.high - region.low + 1 for region in REGIONS)


__all__ = [
    "MotifHit",
    "align_region",
    "region_shifts",
    "ZSCALES_BY_LETTER",
    "charged_positions",
    "find_motif",
    "rank_by_site_similarity",
    "region_identity",
    "site_distance",
    "site_identity",
    "site_length",
]
