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
#: inhibition. The coordinating atom is a pyridine-type aromatic nitrogen (two ring
#: bonds, no hydrogen) whose lone pair points out of the ring; an NH nitrogen is
#: excluded because its lone pair is delocalised into the aromatic system.
#:
#: The earlier definition was a bare ``[nX2;H0]``, which matched any pyridine and so
#: flagged 71.5% of this drug-like library -- useless as a type-II descriptor. It is
#: now restricted to the azole motif (a five-membered aromatic ring with at least two
#: nitrogens: imidazole, triazole, tetrazole), which is the strong type-II coordinator
#: seen in the azole antifungals. That is 4.3% of the library, and it is measurably
#: harder for a fingerprint model: mean absolute error 0.712 on those compounds against
#: 0.512 on the rest (Mann-Whitney p=0.028, three similarity-matched splits pooled).
#: The gap is the empirical case for a structure model here -- a co-folded affinity head
#: masks the heme out of its pair representation, so this is exactly what it cannot
#: score and what a hand-built flag must supply.

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


def azole_coordinators(molecule: Chem.Mol) -> int:
    """Count coordinating nitrogens in azole rings (the type-II heme motif).

    A coordinating nitrogen is a pyridine-type aromatic N -- two ring bonds, no hydrogen
    -- sitting in a five-membered aromatic ring that holds at least two nitrogens. This
    catches imidazole, triazole and tetrazole whether or not the ring is N-substituted,
    so it sees the azole antifungals (fluconazole, ketoconazole) that an ``[nH]`` pattern
    would miss.
    """
    rings = molecule.GetRingInfo().AtomRings()
    count = 0
    for ring in rings:
        if len(ring) != 5:
            continue
        atoms = [molecule.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetIsAromatic() for a in atoms):
            continue
        if sum(a.GetAtomicNum() == 7 for a in atoms) < 2:
            continue
        count += sum(
            a.GetAtomicNum() == 7 and a.GetTotalNumHs() == 0 and a.GetDegree() == 2 for a in atoms
        )
    return count


def mechanistic(smiles: Sequence[str]) -> pd.DataFrame:
    """Substructure counts chosen for a named binding mechanism, not by search.

    Each column corresponds to an interaction with a specific structural basis, so a
    non-zero coefficient is interpretable rather than merely predictive.
    """
    molecules = parse(smiles)
    patterns = {
        "n_basic_amines": BASIC_AMINE_SMARTS,
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
    frame["n_heme_coordinating_n"] = np.array(
        [azole_coordinators(m) for m in molecules], dtype=np.float32
    )
    frame["has_basic_amine"] = (frame["n_basic_amines"] > 0).astype(np.float32)
    frame["is_type_ii_candidate"] = (frame["n_heme_coordinating_n"] > 0).astype(np.float32)
    frame = pd.concat([frame, basic_amine_geometry(molecules)], axis=1)
    return frame


#: Bond counts separating a basic nitrogen from an aromatic ring in the CYP2D6
#: pharmacophore. The published range is a 5-7 A through-space separation; on a
#: saturated chain that is roughly four to six bonds, so the window is set generously
#: either side rather than pretending the mapping is exact.
PHARMACOPHORE_BONDS = (3, 8)


def basic_amine_geometry(molecules: list[Chem.Mol]) -> pd.DataFrame:
    """How far a basic nitrogen sits from the nearest aromatic ring, in bonds.

    Counting basic amines says CYP2D6 will like a molecule; it does not say the amine
    is arranged the way CYP2D6 needs. The isoform's recognition is an ion pair to
    Asp301 or Glu216 with an aromatic ring stacked in the pocket, so the separation
    between the two matters and a count cannot see it.

    Distances are topological -- shortest path in bonds -- not through-space. That is
    deliberate: conformer-derived 3D shape descriptors were measured on this dataset
    and came out worse than the mean alone, and a bond count needs no conformer, no
    alignment and no random seed.

    **Measured, and the gain is marginal.** Five-fold cross-validated R2 added to the
    existing physicochemical, fingerprint and substructure blocks: CYP2D6 0.1139 to
    0.1193, CYP2C9 0.3543 to 0.3680, CYP1A2 0.1857 to 0.1809, CYP3A4 0.5757 to 0.5729.
    The +0.005 on CYP2D6 is the endpoint these were designed for and is within the
    noise of a single cross-validation. Kept because they are three cheap and
    interpretable columns that do not hurt, not because they were shown to help.

    Returns:
        Three columns. ``min_basic_n_to_aromatic_bonds`` is the shortest such path,
        with -1 where the molecule lacks a basic amine or an aromatic ring;
        ``basic_n_in_pharmacophore_window`` flags a path inside
        :data:`PHARMACOPHORE_BONDS`; ``n_aromatic_rings_near_basic_n`` counts the rings
        reachable within the window.
    """
    pattern = Chem.MolFromSmarts(BASIC_AMINE_SMARTS)
    rows = []
    for molecule in molecules:
        nitrogens = [match[0] for match in molecule.GetSubstructMatches(pattern)]
        aromatics = [a.GetIdx() for a in molecule.GetAtoms() if a.GetIsAromatic()]
        if not nitrogens or not aromatics:
            rows.append((-1.0, 0.0, 0.0))
            continue
        distances = Chem.GetDistanceMatrix(molecule)
        paths = distances[np.ix_(nitrogens, aromatics)]
        shortest = float(paths.min())
        low, high = PHARMACOPHORE_BONDS
        rings = molecule.GetRingInfo().AtomRings()
        near = sum(
            1
            for ring in rings
            if any(molecule.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
            and distances[np.ix_(nitrogens, list(ring))].min() <= high
        )
        rows.append((shortest, float(low <= shortest <= high), float(near)))
    return pd.DataFrame(
        rows,
        columns=[
            "min_basic_n_to_aromatic_bonds",
            "basic_n_in_pharmacophore_window",
            "n_aromatic_rings_near_basic_n",
        ],
        dtype=np.float32,
    )


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
