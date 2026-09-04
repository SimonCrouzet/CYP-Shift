"""Conserved P450 signatures, used to find enzymes rather than to tell them apart.

These motifs are invariant across the superfamily, which rules them out as
conditioning features -- a descriptor built from them would be identical for all eight
of our enzymes. That same invariance is what makes them useful for the opposite job:
deciding whether an unfamiliar sequence is a cytochrome P450 close enough to pretrain
on. A candidate usually arrives as a sequence with no solved structure, so the pocket
descriptors of :mod:`cypshift.data.pocket` cannot be computed for it yet; a signature
scan needs only the sequence.

**The geometry does the work, not the patterns.** Taken alone, ``E[A-Z][A-Z]R`` matches
one to six times in each of our sequences, and the earliest match is near the
N-terminus by chance -- an earlier version of this module ranked genuine P450s at zero
because it required the motifs to appear in fold order and that accidental match broke
it. Measured against the nine reviewed sequences in
:data:`cypshift.data.sequences.ACCESSIONS`:

===============  ==========================  ====================================
motif            occurrences per sequence    residues upstream of the heme loop
===============  ==========================  ====================================
heme loop        exactly 1 in all nine       (anchor)
ExxR             1 to 6                      73 to 77 for the last match
I-helix          1 to 3                      130 to 134 for the last match
PERF             absent from 6 of 9          not used
===============  ==========================  ====================================

So the heme loop anchors the scan and everything else is scored on its distance from
it. The spacing is what discriminates: a random ``E[A-Z][A-Z]R`` lands 75 residues
before the heme loop rarely, and a real helix K does it every time.

The windows are calibrated on human microsomal CYPs of families 1 to 3. That is
deliberate, since the point is to find enzymes similar enough that their data
transfers, not to recognise any P450 -- a bacterial P450 such as P450cam shares the
fold but not the substrate chemistry, and would be a poor pretraining target.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: The heme-binding loop. Its cysteine is the fifth ligand to the heme iron, invariant
#: in every known P450, and its loss abolishes catalysis. Occurs exactly once in each
#: of the nine reviewed sequences, which is what makes it usable as an anchor.
HEME_LOOP = "F[A-Z][A-Z]G[A-Z][A-Z][A-Z]C[A-Z]G"


@dataclass(frozen=True)
class Motif:
    """A signature, and where it sits relative to the heme loop.

    Attributes:
        low: Smallest plausible distance upstream of the heme loop, in residues.
        high: Largest. The measured range widened by a margin, since other families
            vary more than the nine this was calibrated on.
    """

    name: str
    pattern: str
    low: int
    high: int
    weight: float
    role: str

    def matches(self, sequence: str) -> tuple[int, ...]:
        """Zero-based start positions of every match, overlapping included."""
        return tuple(m.start() for m in re.finditer(f"(?=({self.pattern}))", sequence))

    def at_expected_distance(self, sequence: str, anchor: int) -> int | None:
        """Position of the match falling in the window upstream of ``anchor``.

        Returns the closest such match to the anchor, or None if none qualifies.
        """
        found = [p for p in self.matches(sequence) if self.low <= anchor - p <= self.high]
        return max(found) if found else None


#: Calibrated on the nine reviewed human sequences. PERF is deliberately absent: it
#: occurs in only three of them, so requiring it would reject two thirds of the
#: enzymes we already know are P450s.
MOTIFS: tuple[Motif, ...] = (
    Motif(
        "i_helix",
        "[AG]G[A-Z][A-Z]T[TSV]",
        115,
        150,
        2.0,
        "oxygen activation; the acid-alcohol pair that protonates the ferric peroxo",
    ),
    Motif(
        "exxr",
        "E[A-Z][A-Z]R",
        60,
        95,
        2.0,
        "helix K; salt-bridges the meander and locks the heme core",
    ),
)

#: Weight of the anchor itself. Dominant because it is the one truly diagnostic
#: feature: the other two are short patterns made meaningful only by their spacing.
HEME_WEIGHT = 4.0
MAX_SCORE = HEME_WEIGHT + sum(m.weight for m in MOTIFS)


@dataclass(frozen=True)
class SignatureScan:
    """Which P450 signatures a sequence carries, and where."""

    length: int
    heme_loop: int | None
    positions: dict[str, int | None]

    def has(self, name: str) -> bool:
        return self.positions.get(name) is not None

    @property
    def score(self) -> float:
        """Weighted signature completeness, from 0 to 1."""
        if self.heme_loop is None:
            return 0.0
        total = HEME_WEIGHT + sum(m.weight for m in MOTIFS if self.has(m.name))
        return total / MAX_SCORE


def scan(sequence: str) -> SignatureScan:
    """Locate the heme loop, then the other signatures at their expected spacing."""
    anchors = [m.start() for m in re.finditer(f"(?=({HEME_LOOP}))", sequence)]
    if not anchors:
        return SignatureScan(len(sequence), None, {m.name: None for m in MOTIFS})
    anchor = anchors[-1]
    return SignatureScan(
        len(sequence),
        anchor,
        {m.name: m.at_expected_distance(sequence, anchor) for m in MOTIFS},
    )


def p450_score(sequence: str) -> float:
    """How strongly a sequence looks like a drug-metabolising cytochrome P450.

    Not a substitute for an alignment -- a cheap filter deciding what is worth
    aligning.
    """
    return scan(sequence).score


def is_cytochrome_p450(sequence: str, threshold: float = 0.7) -> bool:
    """Whether a sequence is a plausible pretraining relative.

    The default threshold requires the heme loop plus at least one correctly spaced
    signature. The heme loop alone scores 0.5 and does not pass: a single ten-residue
    pattern can occur by chance in a long sequence.
    """
    return p450_score(sequence) >= threshold


def describe(sequence: str) -> str:
    """One line per signature, for reporting why a candidate was accepted."""
    found = scan(sequence)
    lines = [f"{len(sequence)} residues, score {found.score:.2f}"]
    if found.heme_loop is None:
        lines.append("  heme loop  absent -- not a P450")
        return "\n".join(lines)
    lines.append(f"  {'heme_loop':<10} {found.heme_loop + 1:<6} anchor")
    for motif in MOTIFS:
        position = found.positions[motif.name]
        where = (
            f"{position + 1:<6} {found.heme_loop - position} upstream"
            if position is not None
            else f"{'absent':<6} (expected {motif.low}-{motif.high} upstream)"
        )
        lines.append(f"  {motif.name:<10} {where}")
    return "\n".join(lines)
