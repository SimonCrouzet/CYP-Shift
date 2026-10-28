"""Ionization state and through-space pharmacophore geometry at physiological pH.

The mechanistic block already counts basic amines and the *bond* distance from one to an
aromatic ring. Two things it cannot see matter for CYP2D6 (and, by charge, CYP2C9): the
compound's actual charge at pH 7.4, and the *through-space* separation of the protonated
nitrogen from the aromatic it stacks against -- the real CYP2D6 pharmacophore is an
Ångström distance (de Groot), not a bond count.

This block supplies both, computed and cached, never hand-set:

* protonation at pH 7.4 from ``dimorphite-dl`` (its single dominant microstate), giving
  the number of cationic and anionic centres -- so an amine that is actually protonated
  reads as a cation and an acid as an anion, rather than as a neutral SMARTS match;
* a 3D distance from a protonated nitrogen to the nearest aromatic ring centroid, measured
  on a seeded ETKDG conformer. A continuous value, with ``-1`` where the compound has no
  cation or no aromatic ring (the sentinel the mechanistic block already uses); no distance
  threshold is imposed, so the model learns any window itself.

Conformer embedding is the slow step, so results are cached per canonical SMILES to a
parquet and computed once. The block is a swept ``featurise`` hook, off unless requested.
"""

import contextlib
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from cypshift.data.chembl import canonical

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

DEFAULT_CACHE = Path("data/external/ionization/features.parquet")
IONIZATION_COLUMNS = ("n_cationic_ph74", "n_anionic_ph74", "cation_to_aromatic_angstroms")
#: ETKDG seed, so the conformer-derived distance is reproducible.
_EMBED_SEED = 0xF00D
_MISSING_DISTANCE = -1.0
#: Protonation pH and precision used for the dominant microstate.
_PH = 7.4
_PRECISION = 0.0


def _generation_manifest() -> dict[str, object]:
    """What the cached values depend on, so a settings change is detectable as stale."""
    return {
        "tool": "dimorphite-dl",
        "ph": _PH,
        "precision": _PRECISION,
        "embed_seed": _EMBED_SEED,
        "columns": list(IONIZATION_COLUMNS),
    }


def _dominant_microstate(smiles: str) -> str:
    """The single most likely protonation state at pH 7.4, or the input if none is found."""
    from dimorphite_dl import protonate_smiles

    try:
        states = protonate_smiles(smiles, ph_min=_PH, ph_max=_PH, precision=_PRECISION)
    except Exception:  # noqa: BLE001 -- a protonation failure falls back to the input
        return smiles
    return states[0] if states else smiles


def _cation_to_aromatic_angstroms(mol: Chem.Mol) -> float:
    """Shortest 3D distance from a protonated nitrogen to an aromatic ring centroid."""
    cations = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0
    ]
    rings = [
        ring
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    ]
    if not cations or not rings:
        return _MISSING_DISTANCE
    molecule = Chem.AddHs(mol)  # heavy-atom indices are preserved
    params = AllChem.ETKDGv3()
    params.randomSeed = _EMBED_SEED
    if AllChem.EmbedMolecule(molecule, params) != 0:
        return _MISSING_DISTANCE
    # An unparameterised atom leaves the raw ETKDG conformer, which is still usable.
    with contextlib.suppress(Exception):
        AllChem.MMFFOptimizeMolecule(molecule)
    positions = molecule.GetConformer().GetPositions()
    return min(
        float(np.linalg.norm(positions[cation] - positions[list(ring)].mean(axis=0)))
        for cation in cations
        for ring in rings
    )


def _compute_one(smiles: str) -> tuple[float, float, float]:
    mol = Chem.MolFromSmiles(_dominant_microstate(smiles))
    if mol is None:
        return 0.0, 0.0, _MISSING_DISTANCE
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    n_cationic = float(sum(charge > 0 for charge in charges))
    n_anionic = float(sum(charge < 0 for charge in charges))
    return n_cationic, n_anionic, _cation_to_aromatic_angstroms(mol)


def _ensure_cache(canonical_smiles: list[str], cache_path: Path) -> pd.DataFrame:
    """The feature table for the requested compounds, computing and caching any missing."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path.with_suffix(".manifest.json")
    columns = ["SMILES", *IONIZATION_COLUMNS]
    if cache_path.exists():
        # The values depend on the protonation and conformer settings; warn if a cache was
        # generated under different ones, since equal SMILES would otherwise reuse them.
        stored = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if stored != _generation_manifest():
            logger.warning(
                "ionization cache at %s was generated under different settings (%s); its values "
                "may be stale -- delete it to recompute",
                cache_path,
                stored,
            )
        table = pd.read_parquet(cache_path)
    else:
        table = pd.DataFrame(columns=columns)
    missing = sorted({s for s in canonical_smiles if s and s not in set(table["SMILES"])})
    if missing:
        rows = [(s, *_compute_one(s)) for s in missing]
        table = pd.concat([table, pd.DataFrame(rows, columns=columns)], ignore_index=True)
        # Publish atomically: a crash or a concurrent sweep agent mid-write must never leave
        # a truncated parquet that later reads as an apparently valid (but corrupt) cache.
        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
        table.to_parquet(temporary, index=False)
        os.replace(temporary, cache_path)
        manifest_path.write_text(json.dumps(_generation_manifest(), indent=2) + "\n")
        logger.info("ionization: computed %d new compounds (%d cached)", len(missing), len(table))
    return table.set_index("SMILES")


def ionization_features(
    smiles: Sequence[str], cache_path: Path = DEFAULT_CACHE
) -> NDArray[np.float32]:
    """The ionization/geometry block for each compound, in input order."""
    canonical_smiles = list(canonical(pd.Series(list(smiles))))
    table = _ensure_cache(canonical_smiles, cache_path)
    default = np.array([0.0, 0.0, _MISSING_DISTANCE], dtype=float)
    rows = [
        table.loc[c, list(IONIZATION_COLUMNS)].to_numpy(float) if c in table.index else default
        for c in canonical_smiles
    ]
    return np.vstack(rows).astype(np.float32)


def ionization_block(
    cache_path: Path = DEFAULT_CACHE,
) -> Callable[[Sequence[str]], NDArray[np.float32]]:
    """A ``featurise``-compatible callable returning the ionization/geometry block."""

    def block(smiles: Sequence[str]) -> NDArray[np.float32]:
        return ionization_features(smiles, cache_path)

    return block


def block_for(
    enabled: bool, cache_path: Path = DEFAULT_CACHE
) -> Callable[[Sequence[str]], NDArray[np.float32]] | None:
    """The ionization block when the feature is on, or None when off."""
    return ionization_block(cache_path) if enabled else None
