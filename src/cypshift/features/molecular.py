"""Molecular featurisation.

Three blocks, kept separate because they behave very differently across isoforms.
Measured by cross-validated R2 on the challenge training set:

===========  ========  ========  ========  ========
block        CYP1A2    CYP2C9    CYP2D6    CYP3A4
===========  ========  ========  ========  ========
physchem       0.138     0.261     0.039     0.375
fingerprint    0.131     0.103     0.109     0.289
both           0.210     0.311     0.136     0.494
===========  ========  ========  ========  ========

The two blocks swap places between isoforms. CYP2C9 and CYP3A4 ride on bulk
physicochemistry; CYP2D6 does not use it at all, and its signal is substructural.
That is why both blocks are kept and why the mechanistic block exists.
"""

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

#: Aliphatic nitrogen basic enough to be protonated at physiological pH. Worth
#: +0.55 log units on CYP2D6 and negative on the other three isoforms: the Asp301
#: and Glu216 ion pair described in the CYP2D6 crystal structures.
BASIC_AMINE_SMARTS = "[NX3;H2,H1,H0;!$(N[!#6]);!$(N*=[!#6]);!$(Nc);!$(N#*)]"

#: Pyridine-type aromatic nitrogen: two ring connections and no hydrogen, so its lone
#: pair sits in the ring plane and is free to coordinate the heme iron. This is type-II
#: inhibition. An imidazole NH nitrogen is deliberately excluded, because its lone pair
#: is delocalised into the aromatic system and cannot bind the metal.
#:
#: This is precisely the interaction a co-folded affinity head cannot score, since
#: non-polymer cofactors are masked out of its pair representation before the head
#: sees them, which makes it the highest-value hand-built feature to pair with a
#: structure model.
HEME_COORDINATING_SMARTS = "[nX2;H0]"

CARBOXYLIC_ACID_SMARTS = "[CX3](=O)[OX2H1,OX1-]"

PHYSCHEM_FUNCTIONS = {
    "molecular_weight": Descriptors.MolWt,
    "clogp": Descriptors.MolLogP,
    "tpsa_angstrom2": Descriptors.TPSA,
    "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "n_hbond_donors": rdMolDescriptors.CalcNumHBD,
    "n_hbond_acceptors": rdMolDescriptors.CalcNumHBA,
    "n_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "n_heavy_atoms": lambda molecule: molecule.GetNumHeavyAtoms(),
}


def parse(smiles: Sequence[str]) -> list[Chem.Mol]:
    """Parse SMILES, raising on the first failure rather than dropping it silently."""
    molecules = []
    for index, item in enumerate(smiles):
        molecule = Chem.MolFromSmiles(item)
        if molecule is None:
            raise ValueError(f"could not parse SMILES at position {index}: {item!r}")
        molecules.append(molecule)
    return molecules


def morgan_counts(
    smiles: Sequence[str], radius: int = 2, n_bits: int = 2048
) -> NDArray[np.float32]:
    """ECFP4-equivalent count fingerprints, shape ``(n_compounds, n_bits)``."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(
        [generator.GetCountFingerprintAsNumPy(molecule) for molecule in parse(smiles)],
        dtype=np.float32,
    )


def physicochemical(smiles: Sequence[str]) -> pd.DataFrame:
    """Bulk physicochemical descriptors, one column per property."""
    molecules = parse(smiles)
    return pd.DataFrame(
        {
            name: [function(molecule) for molecule in molecules]
            for name, function in PHYSCHEM_FUNCTIONS.items()
        },
        dtype=np.float32,
    )


def mechanistic(smiles: Sequence[str]) -> pd.DataFrame:
    """Substructure counts chosen for a named binding mechanism, not by search.

    Each column corresponds to an interaction with a specific structural basis, so a
    non-zero coefficient is interpretable rather than merely predictive.
    """
    molecules = parse(smiles)
    patterns = {
        "n_basic_amines": BASIC_AMINE_SMARTS,
        "n_heme_coordinating_n": HEME_COORDINATING_SMARTS,
        "n_carboxylic_acids": CARBOXYLIC_ACID_SMARTS,
    }
    frame = pd.DataFrame(
        {
            name: [
                len(molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
                for molecule in molecules
            ]
            for name, smarts in patterns.items()
        },
        dtype=np.float32,
    )
    frame["has_basic_amine"] = (frame["n_basic_amines"] > 0).astype(np.float32)
    frame["is_type_ii_candidate"] = (frame["n_heme_coordinating_n"] > 0).astype(np.float32)
    return frame


def featurise(
    smiles: Sequence[str],
    fingerprint: bool = True,
    physchem: bool = True,
    mechanism: bool = True,
    n_bits: int = 2048,
    extra: Callable[[Sequence[str]], NDArray[np.float32]] | None = None,
) -> NDArray[np.float32]:
    """Assemble the requested blocks into one dense matrix.

    Args:
        smiles: Compounds to featurise.
        fingerprint: Include Morgan count fingerprints.
        physchem: Include bulk physicochemical descriptors.
        mechanism: Include the mechanistic substructure counts.
        n_bits: Fingerprint length.
        extra: Optional block appended last, called with the same compounds. This is
            how a multi-fidelity surrogate enters -- as an input feature rather than
            as extra training rows. See :mod:`cypshift.data.surrogate`.

    Returns:
        Shape ``(n_compounds, n_features)``.
    """
    blocks: list[NDArray[np.float32]] = []
    if fingerprint:
        blocks.append(morgan_counts(smiles, n_bits=n_bits))
    if physchem:
        blocks.append(physicochemical(smiles).to_numpy(dtype=np.float32))
    if mechanism:
        blocks.append(mechanistic(smiles).to_numpy(dtype=np.float32))
    if extra is not None:
        appended = np.asarray(extra(smiles), dtype=np.float32)
        if len(appended) != len(smiles):
            raise ValueError(
                f"extra block returned {len(appended)} rows for {len(smiles)} compounds"
            )
        blocks.append(appended.reshape(len(smiles), -1))
    if not blocks:
        raise ValueError("at least one feature block must be requested")
    matrix = np.hstack(blocks)
    logger.info("featurised %d compounds into %d columns", matrix.shape[0], matrix.shape[1])
    return matrix
