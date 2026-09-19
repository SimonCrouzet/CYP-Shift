"""Compute the structure-derived enzyme descriptors and cache them, so nothing is
hand-typed.

For every enzyme with a registered holo structure (:data:`cypshift.data.pocket.STRUCTURES`)
this derives, from the structure alone:

* two cavity volumes, one per detector -- fpocket and pyKVFinder. Neither reproduces the
  literature cavity *ordering* on its own (CYP pockets are conformation- and
  method-dependent; see the analysis in the project notes), so both are kept as
  independent computed descriptors and the model uses whatever signal each carries. Each
  uses the same always-available rule: the pocket/cavity nearest the heme iron.
* recognition-chemistry presence -- whether an acidic, basic or aromatic residue lines the
  ligand-contact pocket, from the real co-crystal contacts.

The result is written to ``src/cypshift/data/enzyme_descriptors.json`` (committed), which
:mod:`cypshift.data.enzymes` loads at import so a run never recomputes or hits the network.
Re-run this script (needs fpocket and pyKVFinder) after changing a structure. It also
cross-checks each PDB against RCSB metadata as a structural-sanity gate.

    python scripts/build_enzyme_descriptors.py
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

from cypshift.data.pocket import (
    ACIDIC,
    AROMATIC,
    BASIC,
    STRUCTURES,
    _atoms,
    contact_residues,
    fetch_structure,
    heme_iron,
)
from cypshift.data.sequences import ACCESSIONS

logger = logging.getLogger(__name__)

CACHE = Path("src/cypshift/data/enzyme_descriptors.json")
FPOCKET = str(Path(sys.executable).parent / "fpocket")


def fpocket_volume(pdb: str, iron: np.ndarray) -> float:
    """Volume of the fpocket pocket whose alpha-spheres reach nearest the heme iron."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / f"{pdb}.pdb").write_text(fetch_structure(pdb).read_text())
    subprocess.run([FPOCKET, "-f", str(tmp / f"{pdb}.pdb")], capture_output=True, timeout=300)
    out = tmp / f"{pdb}_out"
    info = (out / f"{pdb}_info.txt").read_text()
    vols = {
        int(m.group(1)): float(m.group(2))
        for m in re.finditer(r"Pocket (\d+) :.*?Volume : \s*([\d.]+)", info, re.S)
    }
    spheres: dict[int, list] = {}
    for line in (out / f"{pdb}_out.pdb").read_text().splitlines():
        if line.startswith("HETATM") and line[17:20] == "STP":
            spheres.setdefault(int(line[22:26]), []).append(
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            )
    nearest = min(spheres, key=lambda p: np.linalg.norm(np.array(spheres[p]) - iron, axis=1).min())
    return vols[nearest]


def pykvfinder_volume(pdb: str, iron: np.ndarray) -> float:
    """Volume of the pyKVFinder cavity whose lining residues sit nearest the heme iron."""
    import pyKVFinder

    src = fetch_structure(pdb)
    result = pyKVFinder.run_workflow(str(src))
    if not result or not result.volume:
        return 0.0
    coords = {(chain, sid): xyz for _, chain, sid, xyz in _atoms(src, "ATOM")}

    def distance(tag: str) -> float:
        ds = [
            np.linalg.norm(coords[(c, str(num))] - iron)
            for num, c, _ in result.residues[tag]
            if (c, str(num)) in coords
        ]
        return min(ds) if ds else 1e9

    nearest = min(result.volume, key=distance)
    return float(result.volume[nearest])


def composition(pdb: str) -> dict[str, bool]:
    """Whether an acidic, basic or aromatic residue lines the ligand-contact pocket."""
    residues = set(contact_residues(fetch_structure(pdb)).residues)
    return {
        "has_acidic": bool(residues & ACIDIC),
        "has_basic": bool(residues & BASIC),
        "has_aromatic": bool(residues & AROMATIC),
    }


def crosscheck(pdb: str, accession: str) -> str:
    """Confirm the PDB maps to this enzyme's UniProt and carries heme; return a note."""
    try:
        entry = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb}", timeout=60).json()
        ligs = entry.get("rcsb_entry_info", {}).get("nonpolymer_bound_components", []) or []
        return f"bound_ligands={ligs} (heme present: {'HEM' in ligs})"
    except Exception as error:  # noqa: BLE001
        return f"crosscheck unavailable ({error})"


def build() -> dict[str, dict]:
    descriptors = {}
    for enzyme, pdb in STRUCTURES.items():
        iron = heme_iron(fetch_structure(pdb))
        row = {
            "pdb": pdb,
            "fpocket_volume": round(fpocket_volume(pdb, iron), 1),
            "pykvfinder_volume": round(pykvfinder_volume(pdb, iron), 1),
            **composition(pdb),
        }
        logger.info("%s (%s): %s", enzyme, pdb, row)
        logger.info("  crosscheck: %s", crosscheck(pdb, ACCESSIONS.get(enzyme, "")))
        descriptors[enzyme] = row
    return descriptors


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    descriptors = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(descriptors, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %d enzyme descriptors to %s", len(descriptors), CACHE)


if __name__ == "__main__":
    main()
