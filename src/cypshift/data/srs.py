"""Substrate recognition sites: encoding each pocket region separately.

Pooling every contact residue into one mean is what
:func:`cypshift.data.zscales.pooled_zscales` does, and it has a measured failure. The
residue that decides an enzyme's recognition chemistry is usually one or two out of
sixteen to twenty-six, so averaging buries it: CYP2D6's Asp301 and Glu216 barely move
the mean, and the pooled vectors of the four scored enzymes sit within 0.5 to 1.0 of
each other in a space where individual Z-scales span roughly plus or minus four.

Gotoh identified the fix in 1992 (J Biol Chem 267:83). Substrate specificity is not
spread evenly along a P450: it concentrates in six hypervariable segments, about 16% of
the sequence, into which nearly every specificity-altering mutation falls. Encoding
each segment separately keeps the anchor, because the anchor has a segment largely to
itself.

**How the segments are located.** Gotoh's boundaries are given in a CYP2-family
alignment frame anchored on bacterial P450cam, so they do not transfer to CYP1A2 or
CYP3A4 by residue number. Rather than import numbers that do not apply, the segments
are found from the structures: contact residues cluster into contiguous runs, and those
runs land at reproducible distances from the heme-binding loop, which
:mod:`cypshift.data.motifs` locates in any P450 sequence. Measured across the four
scored enzymes, with the run's offset upstream of the heme loop:

==========  ============  ============  ============  ============
offset      CYP1A2        CYP2C9        CYP2D6        CYP3A4
==========  ============  ============  ============  ============
316 to 334  IT, STF       **R108**, VF  LF, FL        **RR**F, SI
211 to 228  TFV           LNIL          AGL**E216**   LLFF, T
191 to 195  F, F          VM            QFL           F
126 to 139  NDGADT        VDGA, T       IA**D301**    IIFA, T
62 to 69    L, I          L, L          V, V          IARLE
around -46  LT            --            F             G
==========  ============  ============  ============  ============

Each of the three residues that define recognition -- CYP2C9's anion-pairing Arg108,
CYP2D6's Glu216 and Asp301 -- falls in a different band. That is the whole argument for
separating them.

CYP3A4 additionally contacts residues 53 to 76, upstream of every band the others use.
This is not an artefact: Zawaira's expanded map (Curr Drug Metab 2011) adds SRS1a and
SRS1b ahead of Gotoh's SRS1 for exactly this reason, raising coverage from 16% to 33%
of the sequence, and it fits CYP3A4's much larger cavity.

The bands are the observed contact spans widened by four residues on each side. That
puts them across 30% of the chain, against Gotoh's 16% for the original six segments
and Zawaira's 33% for the expanded map -- an independent measurement from four
structures landing where the published map says it should.

**A structure is needed to find the bands, not to use them.** Because a band is an
offset from the heme loop and :func:`cypshift.data.motifs.scan` locates that loop in
any P450 sequence, the binding site can be read off a bare sequence once the offsets
are known. :func:`region_sequences` does that, which is what lets an auxiliary enzyme
with no co-crystal still be described and compared.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cypshift.data.motifs import scan
from cypshift.data.pocket import PocketComposition
from cypshift.data.zscales import residue_zscales, sequence_zscales

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """One substrate recognition site, placed by its distance from the heme loop.

    Attributes:
        low: Smallest offset upstream of the heme loop, in residues. Negative for a
            segment lying C-terminal to the loop, as SRS6 does.
        high: Largest such offset.
    """

    name: str
    low: int
    high: int
    element: str

    def contains(self, offset: int) -> bool:
        return self.low <= offset <= self.high


#: Ordered N-terminus to C-terminus, which means decreasing offset from the heme loop.
#: Bands are the measured ranges widened by a margin, since other CYP families vary
#: more than the four these were calibrated on.
REGIONS: tuple[Region, ...] = (
    Region("SRS1a", 355, 386, "beta1 sheet and A helix; only CYP3A4 reaches this far"),
    Region("SRS1", 310, 338, "B' helix and the B'-C loop"),
    Region("SRS2", 207, 232, "C-terminal end of helix F"),
    Region("SRS3", 184, 199, "N-terminal region of helix G"),
    Region("SRS4", 122, 143, "central helix I, the groove directly above the heme"),
    Region("SRS5", 57, 73, "beta1-4 strand, downstream of the ExxR motif"),
    Region("SRS6", -51, -42, "beta4 hairpin, C-terminal to the heme loop"),
)

REGION_NAMES = tuple(region.name for region in REGIONS)

#: Three Z-scales per region. Eighteen to twenty-one dimensions is already wide for
#: eight enzymes, so this is a candidate to test with
#: :func:`cypshift.eval.transfer.leave_one_target_out`, not an automatic default.
SRS_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(
    f"{region.name}_zscale_{index + 1}" for region in REGIONS for index in range(3)
)


def region_for(offset: int) -> Region | None:
    """Which recognition site a residue at ``offset`` upstream of the heme loop is in."""
    for region in REGIONS:
        if region.contains(offset):
            return region
    return None


def assign_regions(pocket: PocketComposition, sequence: str) -> dict[str, tuple[str, ...]]:
    """Group a pocket's contact residues by recognition site.

    Args:
        pocket: Contacts carrying residue numbers, from
            :func:`cypshift.data.pocket.contact_residues`.
        sequence: The enzyme's sequence, used only to locate the heme loop anchor.

    Returns:
        Residue names per region name. Regions with no contact are absent, and
        residues falling outside every band are dropped.

    Raises:
        ValueError: If the pocket carries no residue numbers, or the sequence has no
            heme-binding loop to anchor on.
    """
    if not pocket.numbers:
        raise ValueError("pocket carries no residue numbers to place in a region")
    anchor = scan(sequence).heme_loop
    if anchor is None:
        raise ValueError("sequence has no heme-binding loop to anchor regions on")

    grouped: dict[str, list[str]] = {}
    unassigned = 0
    for name, number in zip(pocket.residues, pocket.numbers, strict=True):
        region = region_for(anchor + 1 - number)
        if region is None:
            unassigned += 1
            continue
        grouped.setdefault(region.name, []).append(name)
    if unassigned:
        logger.info("%s: %d contacts fell outside every region", pocket.pdb_id, unassigned)
    return {name: tuple(residues) for name, residues in grouped.items()}


def region_sequences(sequence: str) -> dict[str, str]:
    """The binding-site subsequence of each recognition site, from sequence alone.

    No structure is involved. The bands were found using four co-crystals, but a band
    is an offset from the heme loop, and :func:`cypshift.data.motifs.scan` locates that
    loop in any P450 sequence -- so an auxiliary enzyme with no solved structure can
    still have its binding site read off.

    Returns:
        Subsequence per region name, in N-to-C order. A region running off either end
        of the sequence is truncated to what exists.

    Raises:
        ValueError: If the sequence carries no heme-binding loop to anchor on.
    """
    anchor = scan(sequence).heme_loop
    if anchor is None:
        raise ValueError("sequence has no heme-binding loop to anchor regions on")
    position = anchor + 1
    extracted: dict[str, str] = {}
    for region in REGIONS:
        start = max(position - region.high - 1, 0)
        stop = min(position - region.low, len(sequence))
        if start < stop:
            extracted[region.name] = sequence[start:stop]
    return extracted


def binding_site(sequence: str) -> str:
    """Every recognition site concatenated, N to C.

    About 30% of the chain. This is the part worth comparing between enzymes: the
    conserved scaffold is identical across the superfamily and contributes nothing to
    a similarity that is meant to predict shared ligand preference.
    """
    return "".join(region_sequences(sequence).values())


def srs_descriptor_from_sequence(sequence: str) -> NDArray[np.float32]:
    """Mean Z-scales per recognition site, needing no structure.

    The counterpart to :func:`srs_descriptor`. It averages over the whole band rather
    than over the residues a ligand was seen to touch, so it is the blunter of the two
    -- but it is defined for every enzyme, which is what pretraining on auxiliaries
    requires.
    """
    extracted = region_sequences(sequence)
    blocks = []
    for region in REGIONS:
        table = sequence_zscales(extracted.get(region.name, ""))
        blocks.append(table.mean(axis=0) if len(table) else np.zeros(3))
    return np.concatenate(blocks).astype(np.float32)


def srs_descriptor(pocket: PocketComposition, sequence: str) -> NDArray[np.float32]:
    """Mean Z-scales per recognition site, in :data:`SRS_DESCRIPTOR_NAMES` order.

    A region with no contact residue is left at zero. That is a real statement about
    the pocket rather than a missing value -- it says the ligand does not reach that
    part of the enzyme, which is how CYP1A2's narrow slot differs from CYP3A4's cavity.
    """
    grouped = assign_regions(pocket, sequence)
    blocks = []
    for region in REGIONS:
        residues = grouped.get(region.name, ())
        table = residue_zscales(residues)
        blocks.append(table.mean(axis=0) if len(table) else np.zeros(3))
    return np.concatenate(blocks).astype(np.float32)
