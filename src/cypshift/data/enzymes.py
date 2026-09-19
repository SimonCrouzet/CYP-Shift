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

**Provenance.** No descriptor value is hand-typed. The two cavity volumes and the pocket
composition are computed from each enzyme's holo crystal structure by
``scripts/build_enzyme_descriptors.py`` (fpocket and pyKVFinder for the volumes, the real
co-crystal contacts for the composition) and cached in ``enzyme_descriptors.json`` beside
this module. Two detectors are kept because neither reproduces the literature cavity
*ordering* on its own -- CYP pockets are conformation- and method-dependent -- so each is
an independent computed signal, dependable as an ordering rather than to three figures.
Only the CYP family number (from the gene name) is definitional. Nothing is inferred from
the challenge data, so a descriptor cannot leak a label.
"""

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

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
    family: int
    #: Two cavity volumes, one per detector (fpocket, pyKVFinder), computed from the holo
    #: structure by scripts/build_enzyme_descriptors.py. Neither reproduces the literature
    #: ordering alone -- CYP cavities are conformation- and method-dependent -- so both are
    #: kept and the model uses whatever signal each carries. Ordinal, not exact.
    fpocket_volume: float
    pykvfinder_volume: float
    #: Whether an acidic / basic / aromatic residue lines the ligand-contact pocket,
    #: computed from the real co-crystal contacts. (Aromatic is present in every CYP
    #: pocket, so that column is effectively constant -- kept for a uniform, computed
    #: vector rather than a hand-picked one.)
    has_acidic: bool
    has_basic: bool
    has_aromatic: bool
    reference: str = ""

    def shape(self) -> list[float]:
        """Geometry: the log of each detector's cavity volume, so a wide range does not
        dominate the binary flags."""
        return [
            float(np.log10(self.fpocket_volume)),
            float(np.log10(self.pykvfinder_volume + 1.0)),
        ]

    def motif(self) -> list[float]:
        """Recognition chemistry lining the pocket, computed from the structure."""
        return [float(self.has_acidic), float(self.has_basic), float(self.has_aromatic)]

    def biochemical(self, families: tuple[int, ...] = (1, 2, 3)) -> list[float]:
        """Sequence family, one-hot with an 'other' slot for anything outside it."""
        encoding = [float(self.family == f) for f in families]
        return [*encoding, float(self.family not in families)]


#: The only definitional facts, kept by hand: CYP family number (from the gene name) and
#: how far each relative sits from a scored enzyme. Everything numeric -- the two cavity
#: volumes and the pocket composition -- is computed from the holo structure by
#: scripts/build_enzyme_descriptors.py and loaded from the cache below, so no descriptor
#: value is hand-typed.
_FAMILY: dict[str, int] = {
    "CYP1A2": 1, "CYP2C9": 2, "CYP2D6": 2, "CYP3A4": 3,  # scored
    "CYP2C19": 2, "CYP1A1": 1, "CYP1B1": 1, "CYP2C8": 2, "CYP2B6": 2, "CYP3A5": 3,
    "CYP3A7": 3, "CYP2C18": 2, "CYP3A43": 3, "CYP2E1": 2, "CYP2A6": 2,  # relatives (<=3.0)
}
_DESCRIPTOR_CACHE = Path(__file__).with_name("enzyme_descriptors.json")

#: The four scored enzymes plus auxiliaries within binding-site distance 3.0.
ENZYMES: dict[str, Enzyme] = {}


def register(enzyme: Enzyme) -> Enzyme:
    """Add an enzyme. No model changes when a new one appears."""
    ENZYMES[enzyme.name] = enzyme
    return enzyme


def _load_registry() -> None:
    """Build ENZYMES from the committed, structure-derived descriptor cache."""
    computed = json.loads(_DESCRIPTOR_CACHE.read_text())
    for name, family in _FAMILY.items():
        if name not in computed:
            raise KeyError(
                f"{name} missing from {_DESCRIPTOR_CACHE.name}; "
                "run scripts/build_enzyme_descriptors.py"
            )
        row = computed[name]
        register(
            Enzyme(
                name=name,
                family=family,
                fpocket_volume=row["fpocket_volume"],
                pykvfinder_volume=row["pykvfinder_volume"],
                has_acidic=row["has_acidic"],
                has_basic=row["has_basic"],
                has_aromatic=row["has_aromatic"],
                reference=f"computed from PDB {row['pdb']}",
            )
        )


_load_registry()


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
