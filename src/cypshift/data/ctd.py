"""Composition and transition descriptors, as used in the published CYP PCM model.

Lapins and Spjuth built a proteochemometric model of CYP inhibition covering our four
scored enzymes plus CYP2C19 (PLoS ONE 2013;8:e66566), reaching AUC 0.923 in
cross-validation and 0.940 on an external set of 8,988 compounds. Their target
descriptors were alignment-independent composition and transition features over seven
amino-acid properties, computed from the primary sequence with PROFEAT -- 21
composition plus 21 transition values, 42 in total. The scheme is Dubchak's (Proteins
1995;23:566).

Reimplemented here rather than called out to, because PROFEAT is a web server and a
descriptor that cannot be recomputed offline cannot be swept.

Each property partitions the twenty amino acids into three classes. *Composition* is
the fraction of residues in each class. *Transition* is how often adjacent residues
cross between two classes, which carries a little sequence order -- the part plain
composition throws away.

**These 42 dimensions describe 8 enzymes.** That is five times more numbers than
things to tell apart, and the model can memorise the mapping without the content
meaning anything, so this is included to reproduce the published setup and to compare
against, not as a default. :mod:`cypshift.eval.transfer` is what decides whether it
earns its width. The narrower alternatives are the Z-scales of
:mod:`cypshift.data.zscales` and the pocket composition of :mod:`cypshift.data.pocket`.
"""

import logging
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Dubchak's seven properties, each partitioning the amino acids into three classes.
#: Order is fixed because it fixes the descriptor's column order.
PROPERTIES: dict[str, tuple[str, str, str]] = {
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    "vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MSPTHY"),
}

#: The three unordered class pairs a transition can cross.
TRANSITIONS = tuple(combinations(range(3), 2))

CTD_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(
    [f"composition_{name}_{index + 1}" for name in PROPERTIES for index in range(3)]
    + [f"transition_{name}_{a + 1}{b + 1}" for name in PROPERTIES for a, b in TRANSITIONS]
)


def _classify(sequence: str, classes: tuple[str, str, str]) -> list[int]:
    """Map each residue to its class index, skipping anything unclassified."""
    lookup = {residue: index for index, group in enumerate(classes) for residue in group}
    return [lookup[residue] for residue in sequence if residue in lookup]


def composition(sequence: str, classes: tuple[str, str, str]) -> NDArray[np.float64]:
    """Fraction of residues falling in each of the three classes."""
    assigned = _classify(sequence, classes)
    if not assigned:
        return np.zeros(3)
    counts = np.bincount(assigned, minlength=3)
    return counts / counts.sum()


def transition(sequence: str, classes: tuple[str, str, str]) -> NDArray[np.float64]:
    """Frequency of adjacent residues crossing between each pair of classes.

    Counted without direction, following Dubchak: a step from class 1 to class 2 and
    one from 2 to 1 are the same transition.
    """
    assigned = _classify(sequence, classes)
    if len(assigned) < 2:
        return np.zeros(len(TRANSITIONS))
    pairs = list(zip(assigned, assigned[1:], strict=False))
    total = len(pairs)
    return np.array(
        [sum((x, y) in {(a, b), (b, a)} for x, y in pairs) / total for a, b in TRANSITIONS]
    )


def ctd_descriptor(sequence: str) -> NDArray[np.float32]:
    """The 42 composition and transition values, in :data:`CTD_DESCRIPTOR_NAMES` order.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not sequence:
        raise ValueError("cannot describe an empty sequence")
    parts = [composition(sequence, c) for c in PROPERTIES.values()]
    parts += [transition(sequence, c) for c in PROPERTIES.values()]
    descriptor = np.concatenate(parts).astype(np.float32)
    if len(descriptor) != len(CTD_DESCRIPTOR_NAMES):
        raise ValueError(f"expected {len(CTD_DESCRIPTOR_NAMES)} values, built {len(descriptor)}")
    return descriptor
