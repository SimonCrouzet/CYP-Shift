"""Computing pocket descriptors from a crystal structure.

The hand-written table in :mod:`cypshift.data.enzymes` encodes what the literature
says about each pocket. This computes descriptors from the structure instead, which
matters for two reasons: the values are reproducible from a PDB identifier rather than
from my reading of a paper, and a new enzyme needs only a structure rather than
someone to summarise one.

The pocket is defined by what a **co-crystallised ligand actually touches**, not by a
sphere around the heme iron. That distinction was measured rather than assumed: a 10 A
shell around the iron captures about 38 residues, and the single anchoring residue that
decides the chemistry is averaged away among them. Descriptors built that way gave
CYP2C9 a net negative charge and CYP2D6 a net positive one -- the reverse of the truth,
since CYP2C9 pairs anions through Arg108 and CYP2D6 pairs protonated amines through
Asp301. Contact residues recover the anchor because they are the surface the ligand is
held by.

A heme-relative shell is still available through ``radius``-based selection, for a
structure with no bound ligand, but it should not be trusted to encode recognition.

Descriptors are compositional rather than geometric. Cavity volume needs a dedicated
detector such as fpocket, and published volumes disagree by method; residue composition
is computable from coordinates alone and is what recognition actually rests on.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from numpy.typing import NDArray

from cypshift.data.zscales import ZSCALE_DESCRIPTOR_NAMES, pooled_zscales

logger = logging.getLogger(__name__)

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DEFAULT_STRUCTURE_DIR = Path("data/structures")

#: Radius around the heme iron, in angstrom, within which a residue is taken to line
#: the active site. Ten covers the substrate cavity of a P450 without reaching into
#: the surrounding fold.
DEFAULT_POCKET_RADIUS = 10.0

ACIDIC = frozenset({"ASP", "GLU"})
BASIC = frozenset({"ARG", "LYS", "HIS"})
AROMATIC = frozenset({"PHE", "TYR", "TRP"})
POLAR = frozenset({"SER", "THR", "ASN", "GLN", "CYS", "TYR"})

#: Kyte-Doolittle hydropathy, the standard scale for how greasy a residue is.
HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

POCKET_DESCRIPTOR_NAMES = (
    "n_lining_residues",
    "fraction_acidic",
    "fraction_basic",
    "fraction_aromatic",
    "fraction_polar",
    "net_charge",
    "mean_hydropathy",
    "mean_distance_to_iron",
)

#: Compositional descriptors followed by the Z-scale summary of the same residues.
FULL_DESCRIPTOR_NAMES = POCKET_DESCRIPTOR_NAMES + ZSCALE_DESCRIPTOR_NAMES


@dataclass(frozen=True)
class PocketComposition:
    """Residues lining an active site, and what they are made of."""

    pdb_id: str
    residues: tuple[str, ...]
    distances: tuple[float, ...]

    def _fraction(self, group: frozenset[str]) -> float:
        return sum(r in group for r in self.residues) / max(len(self.residues), 1)

    def zscale_descriptor(self) -> NDArray[np.float32]:
        """Z-scale summary of the lining residues, from :mod:`cypshift.data.zscales`.

        Complements :meth:`descriptor` rather than replacing it. The compositional
        version counts residues by class; this one keeps the chemistry of each
        identity, so Asp and Glu are no longer the same symbol.
        """
        return pooled_zscales(self.residues)

    def full_descriptor(self) -> NDArray[np.float32]:
        """Compositional and Z-scale descriptors concatenated.

        Fourteen dimensions across eight enzymes, which is already generous: the
        proteochemometrics benchmarks favour few dimensions, and a vector wide enough
        to identify each enzyme is a one-hot index by another name. Prefer this only
        with leave-one-target-out evidence that it transfers.
        """
        return np.concatenate([self.descriptor(), self.zscale_descriptor()])

    def descriptor(self) -> NDArray[np.float32]:
        """Fixed-length vector in the order of :data:`POCKET_DESCRIPTOR_NAMES`."""
        acidic = self._fraction(ACIDIC)
        basic = self._fraction(BASIC)
        hydropathy = [HYDROPATHY.get(r, 0.0) for r in self.residues]
        return np.array(
            [
                len(self.residues),
                acidic,
                basic,
                self._fraction(AROMATIC),
                self._fraction(POLAR),
                (basic - acidic) * len(self.residues),
                float(np.mean(hydropathy)) if hydropathy else 0.0,
                float(np.mean(self.distances)) if self.distances else 0.0,
            ],
            dtype=np.float32,
        )


def fetch_structure(pdb_id: str, directory: Path = DEFAULT_STRUCTURE_DIR) -> Path:
    """Download a PDB entry, reusing a local copy when present."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pdb_id.upper()}.pdb"
    if path.exists():
        return path
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    logger.info("downloading %s", url)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def heme_iron(path: Path) -> NDArray[np.float64]:
    """Coordinates of the heme iron, the anchor the pocket is measured from.

    Raises:
        ValueError: If the structure holds no heme iron, which means it is not a
            cytochrome P450 in a form this can describe.
    """
    for line in path.read_text().splitlines():
        is_heme_iron = (
            line.startswith("HETATM")
            and line[17:20].strip() == "HEM"
            and line[12:16].strip() == "FE"
        )
        if is_heme_iron:
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    raise ValueError(f"{path.name} contains no heme iron")


#: Residues within this distance of a bound ligand atom are in contact with it. Four
#: and a half angstrom is the usual cut for a van der Waals contact.
DEFAULT_CONTACT_RADIUS = 4.5

#: Het codes that are solvent, ions or the cofactor rather than a bound ligand.
NOT_A_LIGAND = frozenset(
    {"HOH", "HEM", "SO4", "PO4", "GOL", "EDO", "CL", "NA", "K", "MG", "ZN", "PEG", "MPD", "ACT"}
)


def _atoms(path: Path, record: str) -> list[tuple[str, str, str, NDArray[np.float64]]]:
    """Parsed ``(residue_name, chain, sequence_id, coordinates)`` for one record type."""
    parsed = []
    for line in path.read_text().splitlines():
        if line.startswith(record):
            parsed.append(
                (
                    line[17:20].strip(),
                    line[21],
                    line[22:27].strip(),
                    np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                )
            )
    return parsed


def bound_ligand_atoms(path: Path, min_atoms: int = 10) -> NDArray[np.float64]:
    """Coordinates of the largest bound ligand, ignoring solvent, ions and the heme.

    Raises:
        ValueError: If the structure holds no ligand big enough to define a pocket.
    """
    groups: dict[tuple[str, str, str], list[NDArray[np.float64]]] = {}
    for name, chain, sequence, position in _atoms(path, "HETATM"):
        if name in NOT_A_LIGAND:
            continue
        groups.setdefault((name, chain, sequence), []).append(position)
    if not groups:
        raise ValueError(f"{path.name} has no bound ligand to define a pocket from")
    key, positions = max(groups.items(), key=lambda item: len(item[1]))
    if len(positions) < min_atoms:
        raise ValueError(f"{path.name}: largest ligand {key[0]} has only {len(positions)} atoms")
    logger.info("%s: pocket defined by ligand %s (%d atoms)", path.stem, key[0], len(positions))
    return np.array(positions)


def contact_residues(path: Path, radius: float = DEFAULT_CONTACT_RADIUS) -> PocketComposition:
    """Residues in van der Waals contact with the bound ligand.

    This is the recognition surface -- the residues holding the ligand in place --
    rather than everything within some distance of the cofactor.
    """
    ligand = bound_ligand_atoms(path)
    nearest: dict[tuple[str, str], tuple[str, float]] = {}
    for name, chain, sequence, position in _atoms(path, "ATOM"):
        if name not in HYDROPATHY:
            continue
        distance = float(np.linalg.norm(ligand - position, axis=1).min())
        if distance > radius:
            continue
        key = (chain, sequence)
        if key not in nearest or distance < nearest[key][1]:
            nearest[key] = (name, distance)
    residues = tuple(name for name, _ in nearest.values())
    distances = tuple(distance for _, distance in nearest.values())
    logger.info("%s: %d residues contacting the ligand", path.stem, len(residues))
    return PocketComposition(path.stem, residues, distances)


def lining_residues(path: Path, radius: float = DEFAULT_POCKET_RADIUS) -> PocketComposition:
    """Residues with any atom within ``radius`` of the heme iron."""
    iron = heme_iron(path)
    nearest: dict[tuple[str, str], tuple[str, float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[17:20].strip()
        if name not in HYDROPATHY:
            continue
        position = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        distance = float(np.linalg.norm(position - iron))
        if distance > radius:
            continue
        key = (line[21], line[22:27].strip())
        if key not in nearest or distance < nearest[key][1]:
            nearest[key] = (name, distance)

    residues = tuple(name for name, _ in nearest.values())
    distances = tuple(distance for _, distance in nearest.values())
    logger.info("%s: %d residues within %.0f A of the iron", path.stem, len(residues), radius)
    return PocketComposition(path.stem, residues, distances)


def describe_structure(
    pdb_id: str,
    directory: Path = DEFAULT_STRUCTURE_DIR,
    radius: float = DEFAULT_CONTACT_RADIUS,
) -> NDArray[np.float32]:
    """Pocket descriptor for one PDB entry, from its bound ligand's contact surface.

    Falls back to a heme-relative shell if the entry has no bound ligand, which is a
    weaker descriptor and says so in the log.
    """
    path = fetch_structure(pdb_id, directory)
    try:
        return contact_residues(path, radius).descriptor()
    except ValueError as reason:
        logger.warning("%s; falling back to a heme shell, which is weaker", reason)
        return lining_residues(path).descriptor()
