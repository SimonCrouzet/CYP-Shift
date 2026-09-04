"""Z-scales: encoding what the pocket residues *are*, not just how many.

The compositional descriptors in :mod:`cypshift.data.pocket` count residues by class,
which throws away two things. Aspartate and glutamate become the same symbol despite
differing in reach, and a residue's identity is reduced to whichever bucket it falls
in. Z-scales keep the chemistry: each amino acid maps to three numbers from a principal
component analysis of 26 measured physicochemical properties (Sandberg et al., J Med
Chem 1998;41:2481), conventionally read as lipophilicity, steric bulk and polarity.

Three dimensions is not an arbitrary truncation. van Westen benchmarked thirteen
protein-descriptor sets for proteochemometrics (J Cheminform 2013;5:41 and 5:42) and
the winner was Z-scales(3) with a per-target average; the higher-dimensional sets,
including their own ProtFP(PCA8), came out worst. That result decides the shape of this
module, because we have eight enzymes and therefore eight distinct conditioning
vectors. A descriptor wide enough to identify each one is a one-hot index wearing a
costume, and the optimiser will treat it as such.

What is deliberately *not* encoded: the conserved P450 motifs. FxxGxxxCxG on the heme
loop, ExxR in helix K, PERF, the I-helix oxygen-activation motif -- all are invariant
across the superfamily, which is what makes them useful for recognising a P450 and
useless for telling two apart. The variable regions are the substrate recognition
sites of Gotoh (J Biol Chem 1992;267:83), about 16% of the sequence, and the residues
lining the pocket are largely drawn from them.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Sandberg et al. 1998, principal components z1-z3 of 26 physicochemical properties.
#: z1 rises with hydrophilicity, z2 with steric bulk, z3 with polarity and charge.
ZSCALES: dict[str, tuple[float, float, float]] = {
    "ALA": (0.24, -2.32, 0.60),
    "ARG": (3.52, 2.50, -3.50),
    "ASN": (3.05, 1.62, 1.04),
    "ASP": (3.98, 0.93, 1.93),
    "CYS": (0.84, -1.67, 3.71),
    "GLN": (1.75, 0.50, -1.44),
    "GLU": (3.11, 0.26, -0.11),
    "GLY": (2.05, -4.06, 0.36),
    "HIS": (2.47, 1.95, 0.26),
    "ILE": (-3.89, -1.73, -1.71),
    "LEU": (-4.28, -1.30, -1.49),
    "LYS": (2.29, 0.89, -2.49),
    "MET": (-2.85, -0.22, 0.47),
    "PHE": (-4.22, 1.94, 1.06),
    "PRO": (-1.66, 0.27, 1.84),
    "SER": (2.39, -1.07, 1.15),
    "THR": (0.75, -2.18, -1.12),
    "TRP": (-4.36, 3.94, 0.59),
    "TYR": (-2.54, 2.44, 0.43),
    "VAL": (-2.59, -2.64, -1.54),
}

ZSCALE_DESCRIPTOR_NAMES = (
    "zscale_1_mean",
    "zscale_2_mean",
    "zscale_3_mean",
    "zscale_1_spread",
    "zscale_2_spread",
    "zscale_3_spread",
)


def residue_zscales(residues: tuple[str, ...]) -> NDArray[np.float64]:
    """Look up each residue, dropping anything not among the twenty standard ones.

    Non-standard residues are skipped rather than zero-filled: a zero vector is a
    meaningful point in Z-scale space, close to glutamine, so filling with it would
    invent chemistry that is not there.
    """
    known = [ZSCALES[r] for r in residues if r in ZSCALES]
    if len(known) < len(residues):
        logger.info("skipped %d non-standard residues", len(residues) - len(known))
    return np.array(known, dtype=np.float64).reshape(-1, 3)


def pooled_zscales(residues: tuple[str, ...]) -> NDArray[np.float32]:
    """Mean and spread of the Z-scales over a set of pocket residues.

    The mean is van Westen's per-target average, which is the part with benchmark
    support. The standard deviation is added because two pockets can average alike
    while differing in whether the character is uniform or split between a polar
    anchor and a greasy wall -- which is precisely how CYP2D6 differs from CYP1A2.

    Returns:
        Six values in the order of :data:`ZSCALE_DESCRIPTOR_NAMES`, or zeros if no
        residue was recognised.
    """
    table = residue_zscales(residues)
    if not len(table):
        return np.zeros(len(ZSCALE_DESCRIPTOR_NAMES), dtype=np.float32)
    return np.concatenate([table.mean(axis=0), table.std(axis=0)]).astype(np.float32)
