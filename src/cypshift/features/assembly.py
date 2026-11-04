"""The one place the shared molecular feature set is assembled.

Every dense model (boosted, neural, curve, tfm) builds its input from the same configured
blocks. Keeping the assembly in a single function means a model can only swap *which model*
it is, never silently acquire a different feature set -- the drift that once left the TFM
without the fingerprint. The graph is the one exception: it reads the molecular graph
directly and takes these dense blocks as conditioning instead.

Model-specific adaptation (the TFM's impute/scale/PCA to fit its feature ceiling, the curve's
block scaler, a model's one-hot conditioning) happens in the model, on top of this shared set.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cypshift.features.embeddings import block_for
from cypshift.features.ionization import block_for as ionization_block_for
from cypshift.features.molecular import featurise


def molecular_feature_matrix(
    smiles: Sequence[str],
    *,
    n_bits: int,
    fingerprint_radius: int,
    physchem: bool,
    mechanism: bool,
    ionization_geometry: bool,
    ionization_cache: str,
    molecular_embedding: str,
    use_fingerprint: bool = True,
    screen_model: object | None = None,
) -> NDArray[np.float32]:
    """The configured dense feature set: fingerprint + physchem + mechanism + screen_model +
    ionization + embedding, in that fixed block order. use_fingerprint drops the Morgan block."""
    return featurise(
        list(smiles),
        fingerprint=use_fingerprint,
        n_bits=n_bits,
        radius=fingerprint_radius,
        physchem=physchem,
        mechanism=mechanism,
        extra=screen_model,
        ionization=ionization_block_for(ionization_geometry, Path(ionization_cache)),
        embedding=block_for(molecular_embedding),
    )
