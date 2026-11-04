"""Cofold a CYP with an inhibitor using Boltz-2, and read features off the result.

The affinity head (A) uses Boltz-2's affinity module: a predicted binding value and a
binary binding probability for the protein-ligand complex. The pose head (B) reads the
predicted structure: the inhibitor is cofolded together with the heme cofactor, so the
distance from its nearest coordinating nitrogen to the heme iron is a direct, structural
read on type-II (heme-coordinating) inhibition -- exactly the mechanism a fingerprint
misses and the one measured to be 40% harder for the ligand model.

Running Boltz needs the ``cypshift-struct`` environment and a GPU; this module shells out
to the ``boltz`` executable rather than importing it, so it imports cleanly anywhere and
the feature-reading side works in the ligand environment too. Results are cached per
(enzyme, canonical SMILES) so the expensive step happens once.
"""

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from cypshift.data.sequences import sequence_for

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/structure")
#: The boltz executable next to the running interpreter, so it is found without PATH.
DEFAULT_BOLTZ = str(Path(sys.executable).parent / "boltz")
HEME_CCD = "HEM"
#: Chain ids assigned in the cofold input.
PROTEIN_CHAIN, HEME_CHAIN, LIGAND_CHAIN = "A", "B", "C"

#: The features a cofold yields, in a fixed order. Affinity (head A) then pose and
#: confidence (head B).
STRUCTURE_FEATURE_NAMES = (
    "affinity_pred_value",
    "affinity_probability_binary",
    "ligand_iron_min_distance",
    "coordinating_n_iron_distance",
    "ligand_protein_contacts",
    "complex_plddt",
    "ligand_iptm",
)


def canonical_key(smiles: str) -> str:
    """Short stable hash of a compound's canonical SMILES, for a cache filename.

    Canonicalising first means ``CCO`` and ``OCC`` hash to one key and share a cached
    cofold; hashing the raw string (as before) missed the cache for equivalent spellings.
    An unparseable SMILES falls back to its raw text rather than failing.
    """
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    key = Chem.MolToSmiles(molecule) if molecule is not None else smiles
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def write_boltz_input(enzyme: str, smiles: str, path: Path, affinity: bool = True) -> Path:
    """Write the Boltz YAML: the CYP, its heme cofactor, and the inhibitor.

    The heme is included as an explicit ``HEM`` ligand so the cofold places the
    inhibitor relative to it -- without the cofactor the coordinating distance would be
    meaningless.
    """
    sequences = [
        {"protein": {"id": PROTEIN_CHAIN, "sequence": sequence_for(enzyme)}},
        {"ligand": {"id": HEME_CHAIN, "ccd": HEME_CCD}},
        {"ligand": {"id": LIGAND_CHAIN, "smiles": smiles}},
    ]
    document: dict = {"version": 1, "sequences": sequences}
    if affinity:
        document["properties"] = [{"affinity": {"binder": LIGAND_CHAIN}}]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def run_boltz(
    input_yaml: Path,
    out_dir: Path,
    boltz: str = DEFAULT_BOLTZ,
    use_msa_server: bool = True,
    devices: int = 1,
    extra: tuple[str, ...] = (),
) -> Path:
    """Invoke ``boltz predict`` and return the per-prediction output directory.

    Raises:
        RuntimeError: If boltz exits non-zero or the expected output is absent.
    """
    command = [
        boltz,
        "predict",
        str(input_yaml),
        "--out_dir",
        str(out_dir),
        "--output_format",
        "pdb",
        "--devices",
        str(devices),
    ]
    if use_msa_server:
        command.append("--use_msa_server")
    command.extend(extra)
    logger.info("running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"boltz failed:\n{result.stderr[-2000:]}")
    name = input_yaml.stem
    predictions = out_dir / f"boltz_results_{name}" / "predictions" / name
    if not predictions.is_dir():
        raise RuntimeError(f"boltz produced no predictions at {predictions}")
    return predictions


def _atoms(pdb: Path) -> list[tuple[str, str, str, np.ndarray]]:
    """(record, residue_name, chain, xyz) for every ATOM/HETATM line."""
    rows = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            rows.append((line[:6].strip(), line[17:20].strip(), line[21], line[12:16].strip(), xyz))
    return (
        [(rec, res, chain, xyz) for rec, res, chain, _name, xyz in rows]
        if False
        else [(rec, res, chain, name, xyz) for rec, res, chain, name, xyz in rows]
    )


def pose_features(pdb: Path) -> dict[str, float]:
    """Structural features of the cofolded pose, relative to the heme iron.

    Returns NaN for a distance when the iron or the ligand is absent, so a malformed
    pose does not masquerade as a contact.
    """
    # Boltz names every CCD/SMILES ligand residue "LIG" in a PDB and distinguishes the
    # heme from the inhibitor only by chain, so identify the iron by its atom name in the
    # heme chain and the inhibitor by its chain rather than by residue name.
    atoms = _atoms(pdb)
    iron = [xyz for _rec, _res, chain, name, xyz in atoms if name == "FE" and chain == HEME_CHAIN]
    ligand = [(name, xyz) for _rec, _res, chain, name, xyz in atoms if chain == LIGAND_CHAIN]
    protein = [xyz for rec, _res, _c, _name, xyz in atoms if rec == "ATOM"]

    if not iron or not ligand:
        return {
            "ligand_iron_min_distance": float("nan"),
            "coordinating_n_iron_distance": float("nan"),
            "ligand_protein_contacts": 0.0,
        }
    fe = iron[0]
    lig_xyz = np.array([xyz for _n, xyz in ligand])
    to_iron = np.linalg.norm(lig_xyz - fe, axis=1)
    # Distance from the nearest ligand nitrogen (any atom whose PDB name starts with N) to
    # the heme iron: a geometric proxy for type-II coordination, not a verified coordinate
    # bond, and it does not distinguish a donor lone pair from a protonated or amide N.
    nitrogens = [xyz for name, xyz in ligand if name.startswith("N")]
    n_iron = (
        float(np.min([np.linalg.norm(xyz - fe) for xyz in nitrogens]))
        if nitrogens
        else float("nan")
    )
    contacts = 0.0
    if protein:
        protein_xyz = np.array(protein)
        contacts = float(
            sum(np.linalg.norm(protein_xyz - xyz, axis=1).min() <= 4.5 for _n, xyz in ligand)
        )
    return {
        "ligand_iron_min_distance": float(to_iron.min()),
        "coordinating_n_iron_distance": n_iron,
        "ligand_protein_contacts": contacts,
    }


def parse_scores(predictions: Path, name: str) -> dict[str, float]:
    """Confidence and affinity scalars from the JSON summaries Boltz writes."""
    scores: dict[str, float] = {}
    confidence = predictions / f"confidence_{name}_model_0.json"
    if confidence.exists():
        data = json.loads(confidence.read_text())
        scores["complex_plddt"] = float(data.get("complex_plddt", float("nan")))
        scores["ligand_iptm"] = float(data.get("ligand_iptm", float("nan")))
    affinity = predictions / f"affinity_{name}.json"
    if affinity.exists():
        data = json.loads(affinity.read_text())
        scores["affinity_pred_value"] = float(data.get("affinity_pred_value", float("nan")))
        scores["affinity_probability_binary"] = float(
            data.get("affinity_probability_binary", float("nan"))
        )
    return scores


def features_from_predictions(predictions: Path, name: str) -> dict[str, float]:
    """Assemble the full feature dict from a finished prediction directory."""
    pdb = predictions / f"{name}_model_0.pdb"
    features = dict.fromkeys(STRUCTURE_FEATURE_NAMES, float("nan"))
    if pdb.exists():
        features.update(pose_features(pdb))
    features.update(parse_scores(predictions, name))
    return {k: features.get(k, float("nan")) for k in STRUCTURE_FEATURE_NAMES}


def cofold_features(
    enzyme: str,
    smiles: str,
    cache: Path = DEFAULT_CACHE,
    boltz: str = DEFAULT_BOLTZ,
    affinity: bool = True,
    **run_kwargs: object,
) -> dict[str, float]:
    """Cofold one (enzyme, inhibitor) pair and return its features, cached.

    A cache hit returns immediately; otherwise Boltz runs and the result is stored. The
    cache key is the enzyme and the canonical-SMILES hash.
    """
    target_dir = cache / enzyme
    target_dir.mkdir(parents=True, exist_ok=True)
    cached = target_dir / f"{canonical_key(smiles)}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    work = target_dir / canonical_key(smiles)
    work.mkdir(exist_ok=True)
    input_yaml = write_boltz_input(
        enzyme, smiles, work / f"{enzyme}_{canonical_key(smiles)}.yaml", affinity
    )
    predictions = run_boltz(input_yaml, work, boltz=boltz, **run_kwargs)
    features = features_from_predictions(predictions, input_yaml.stem)
    features["smiles"] = smiles
    cached.write_text(json.dumps(features, indent=2))
    return features
