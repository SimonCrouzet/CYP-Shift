"""Describing an enzyme rather than indexing it.

A one-hot says CYP2C9 and CYP2D6 are two different symbols. It cannot say that one
recognises anions and the other cations, which is *why* a carboxylic acid helps at the
first and hurts at the second, and why a basic amine does the reverse. A model given
the symbol has to learn each enzyme's chemistry separately from that enzyme's own
data; a model given the chemistry can learn a rule -- an acidic anchor in the pocket
prefers a basic ligand -- and apply it to an enzyme it has never been scored on.

That is what makes pretraining across enzymes possible. Under per-enzyme heads,
CYP2C19's 14,492 public measurements would train parameters that are discarded at
prediction time. Described instead, they train the shared mapping from pocket
chemistry to ligand preference, and since CYP2C9 and CYP2C19 are about 91% identical
their descriptors nearly coincide, so that data pools almost directly into a scored
endpoint.

Adding an enzyme is :func:`register`, not a change to any model.

**Provenance.** Pocket volumes are literature values obtained by different cavity
detection methods and are dependable as an *ordering*, not to three significant
figures. Recognition motifs come from the crystallographic descriptions cited per
entry. Family assignment is definitional. Nothing here is inferred from the challenge
data, so a descriptor cannot leak a label.
"""

import logging
from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Which families of descriptor to build. ``shape`` is pocket geometry, ``motif`` the
#: recognition chemistry that decides which ligands fit, ``biochemical`` the sequence
#: family a protein belongs to.
DESCRIPTOR_FAMILIES = ("shape", "motif", "biochemical")


@dataclass(frozen=True)
class Enzyme:
    """What a model needs to know about an enzyme to predict ligands for it.

    Attributes:
        name: Gene symbol.
        pocket_volume_angstrom3: Active-site cavity volume. Ordinal, not exact.
        has_acidic_anchor: An Asp or Glu positioned to ion-pair a protonated ligand
            nitrogen. This is what makes an enzyme prefer basic amines.
        has_basic_anchor: An Arg or Lys positioned to pair with a ligand anion, which
            is what makes an enzyme prefer acids.
        is_narrow_planar: A compact slot that accepts flat, rigid aromatics and little
            else.
        binds_multiple_ligands: Crystallographically observed with more than one
            ligand bound at once, which implies a large or plastic cavity.
        family: CYP family number, 1, 2, 3 and so on.
        reference: Where the structural claims come from.
    """

    name: str
    pocket_volume_angstrom3: float
    has_acidic_anchor: bool
    has_basic_anchor: bool
    is_narrow_planar: bool
    binds_multiple_ligands: bool
    family: int
    reference: str = ""

    def shape(self) -> list[float]:
        """Geometry: log volume, so a 4x range does not dominate the binary flags."""
        return [float(np.log10(self.pocket_volume_angstrom3))]

    def motif(self) -> list[float]:
        """Recognition chemistry: what the pocket is set up to bind."""
        return [
            float(self.has_acidic_anchor),
            float(self.has_basic_anchor),
            float(self.is_narrow_planar),
            float(self.binds_multiple_ligands),
        ]

    def biochemical(self, families: tuple[int, ...] = (1, 2, 3)) -> list[float]:
        """Sequence family, one-hot with an 'other' slot for anything outside it."""
        encoding = [float(self.family == f) for f in families]
        return [*encoding, float(self.family not in families)]


#: The four scored enzymes plus auxiliaries with enough public data to pretrain on.
ENZYMES: dict[str, Enzyme] = {}


def register(enzyme: Enzyme) -> Enzyme:
    """Add an enzyme. No model changes when a new one appears."""
    ENZYMES[enzyme.name] = enzyme
    return enzyme


register(
    Enzyme("CYP1A2", 375, False, False, True, False, 1, "Sansen 2007, JBC 282:14348, PDB 2HI4")
)
register(
    Enzyme("CYP2C9", 470, False, True, False, True, 2, "Williams 2003, Nature 424:464, PDB 1OG5")
)
register(Enzyme("CYP2D6", 540, True, False, False, True, 2, "Rowland 2006, JBC 281:7614, PDB 2F9Q"))
register(
    Enzyme(
        "CYP3A4", 1386, False, False, False, True, 3, "Yano 2004, JBC; Ekroos 2006, PNAS 103:13682"
    )
)

# Auxiliaries. CYP2C19 matters most: about 91% identical to CYP2C9, same family and
# the same anion-recognising architecture, with more public data than CYP2C9 itself.
register(Enzyme("CYP2C19", 470, False, True, False, True, 2, "same subfamily as CYP2C9"))
register(Enzyme("CYP1A1", 375, False, False, True, False, 1, "family 1 planar slot"))
register(Enzyme("CYP1B1", 375, False, False, True, False, 1, "family 1 planar slot"))
register(Enzyme("CYP2C8", 1438, False, True, False, True, 2, "Schoch 2004, JBC 279:9497, PDB 1PQ2"))
register(Enzyme("CYP2B6", 1600, False, False, False, True, 2, "Gay 2010, plastic cavity"))


def descriptor(name: str, families: tuple[str, ...] = DESCRIPTOR_FAMILIES) -> NDArray[np.float32]:
    """Descriptor vector for one enzyme.

    Args:
        name: Gene symbol, which must be registered.
        families: Which descriptor families to include.

    Raises:
        KeyError: If the enzyme is not registered.
        ValueError: On an unknown descriptor family.
    """
    if name not in ENZYMES:
        raise KeyError(f"{name} is not registered; call register() first")
    enzyme = ENZYMES[name]
    values: list[float] = []
    for group in families:
        if group not in DESCRIPTOR_FAMILIES:
            raise ValueError(f"unknown descriptor family {group!r}")
        values.extend(getattr(enzyme, group)())
    return np.array(values, dtype=np.float32)


def descriptor_names(families: tuple[str, ...] = DESCRIPTOR_FAMILIES) -> tuple[str, ...]:
    """Column names the curated table contributes, in order.

    Named per family and position rather than per attribute, because a family is free
    to change how many values it emits without every caller needing to know.
    """
    reference = next(iter(ENZYMES.values()))
    names: list[str] = []
    for group in families:
        if group not in DESCRIPTOR_FAMILIES:
            raise ValueError(f"unknown descriptor family {group!r}")
        names.extend(f"{group}_{index + 1}" for index in range(len(getattr(reference, group)())))
    return tuple(names)


def descriptor_width(families: tuple[str, ...] = DESCRIPTOR_FAMILIES) -> int:
    """Width of the descriptor a family selection produces."""
    return len(descriptor(next(iter(ENZYMES)), families))


def describe_all(
    names: list[str], families: tuple[str, ...] = DESCRIPTOR_FAMILIES
) -> NDArray[np.float32]:
    """Descriptors for a list of enzymes, one row each."""
    return np.stack([descriptor(name, families) for name in names])


def summary() -> str:
    """A readable table of what is registered, for logs and notebooks."""
    columns = [f.name for f in fields(Enzyme) if f.name not in ("name", "reference")]
    lines = ["  ".join([f"{'enzyme':10s}", *(f"{c[:14]:>14s}" for c in columns)])]
    for enzyme in ENZYMES.values():
        cells = [f"{getattr(enzyme, c)!s:>14s}" for c in columns]
        lines.append("  ".join([f"{enzyme.name:10s}", *cells]))
    return "\n".join(lines)
