"""Assembling an enzyme's conditioning vector from the available descriptor sources.

Five sources now describe an enzyme, and they differ in what they need and how wide
they are. This picks among them, so the choice is a swept parameter rather than a
decision buried in a model.

============  =====  =============  ==========================================
source        dims   needs          what it encodes
============  =====  =============  ==========================================
``curated``   9      nothing        the hand-written table, one citation each
``pocket``    8      a structure    residue counts and fractions
``zscales``   6      a structure    Z-scales pooled over all contacts
``srs``       21     both           Z-scales per site, over ligand contacts
``site``      21     a sequence     Z-scales per site, over the whole band
``ctd``       42     a sequence     the published CYP PCM descriptors
============  =====  =============  ==========================================

**Width is the thing to watch.** Eight enzymes means the model ever sees eight distinct
vectors, so a descriptor wide enough to identify each one is a one-hot index in
disguise -- it fits, and no validation curve reveals the difference.
:func:`cypshift.eval.transfer.leave_one_target_out` is what tells them apart, and every
source above ``curated`` should earn its width there before being switched on.

The sources also differ in reach, which matters because pretraining on auxiliary
enzymes is the point of describing an enzyme at all. ``curated``, ``ctd`` and ``site``
need only a table row or a UniProt accession, so they cover every registered enzyme.
``pocket``, ``zscales`` and ``srs`` cover the four scored enzymes, since the
auxiliaries have no comparable co-crystal here.

``site`` and ``srs`` are the same encoding computed two ways, and the pair is worth
understanding. ``srs`` averages over the residues a ligand was actually seen to touch,
which is sharper but needs a co-crystal. ``site`` averages over the whole band, which
is blunter -- it includes residues facing away from the cavity -- but is defined for
any P450 sequence. Structures were needed to find where the bands are; they are not
needed to use them. So an auxiliary enzyme can be described and pretrained on without
one, which is the whole reason for describing an enzyme rather than indexing it.
"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cypshift.data import enzymes
from cypshift.data.ctd import CTD_DESCRIPTOR_NAMES, ctd_descriptor
from cypshift.data.pocket import (
    POCKET_DESCRIPTOR_NAMES,
    STRUCTURES,
    contact_residues,
    fetch_structure,
)
from cypshift.data.sequences import sequence_for
from cypshift.data.srs import (
    SRS_DESCRIPTOR_NAMES,
    srs_descriptor,
    srs_descriptor_from_sequence,
)
from cypshift.data.zscales import ZSCALE_DESCRIPTOR_NAMES

logger = logging.getLogger(__name__)

#: Selectable sources, ordered narrowest first. ``curated`` is the default because it
#: is the only one whose every dimension was chosen for a stated reason.
DESCRIPTOR_SOURCES = ("curated", "pocket", "zscales", "srs", "site", "ctd")

#: Sources that cannot be computed without a co-crystal structure.
STRUCTURAL_SOURCES = frozenset({"pocket", "zscales", "srs"})


def source_names(source: str) -> tuple[str, ...]:
    """Column names one source contributes, in order."""
    match source:
        case "curated":
            return enzymes.descriptor_names()
        case "pocket":
            return POCKET_DESCRIPTOR_NAMES
        case "zscales":
            return ZSCALE_DESCRIPTOR_NAMES
        case "srs":
            return SRS_DESCRIPTOR_NAMES
        case "site":
            return tuple(name.replace("_zscale_", "_band_zscale_") for name in SRS_DESCRIPTOR_NAMES)
        case "ctd":
            return CTD_DESCRIPTOR_NAMES
        case _:
            raise ValueError(f"unknown descriptor source {source!r}")


def descriptor_names(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Every column a source selection produces, in order."""
    return tuple(name for source in sources for name in source_names(source))


def covered_enzymes(sources: tuple[str, ...]) -> set[str]:
    """Enzymes describable by every source in the selection.

    Selecting a structural source silently narrows which enzymes can be pretrained on,
    so this makes the cost visible before a sweep spends time on it.
    """
    covered = set(enzymes.ENZYMES)
    if STRUCTURAL_SOURCES & set(sources):
        covered &= set(STRUCTURES)
    if {"ctd", "site"} & set(sources):
        covered &= set(enzymes.ENZYMES) & _sequenced()
    return covered


def _sequenced() -> set[str]:
    from cypshift.data.sequences import ACCESSIONS

    return set(ACCESSIONS)


def enzyme_descriptor(name: str, sources: tuple[str, ...] = ("curated",)) -> NDArray[np.float32]:
    """Conditioning vector for one enzyme, concatenated across sources.

    Raises:
        ValueError: On an unknown source, or when a source cannot be computed for this
            enzyme. Failing is deliberate: filling with zeros would place an enzyme at
            a real point in descriptor space rather than at a missing one.
    """
    unknown = set(sources) - set(DESCRIPTOR_SOURCES)
    if unknown:
        raise ValueError(f"unknown descriptor sources {sorted(unknown)}")

    blocks: list[NDArray[np.float32]] = []
    pocket = None
    if STRUCTURAL_SOURCES & set(sources):
        if name not in STRUCTURES:
            raise ValueError(f"{name} has no registered structure; drop the structural sources")
        pocket = contact_residues(fetch_structure(STRUCTURES[name]))

    for source in sources:
        match source:
            case "curated":
                blocks.append(enzymes.descriptor(name))
            case "pocket":
                blocks.append(pocket.descriptor())
            case "zscales":
                blocks.append(pocket.zscale_descriptor())
            case "srs":
                blocks.append(srs_descriptor(pocket, sequence_for(name)))
            case "site":
                blocks.append(srs_descriptor_from_sequence(sequence_for(name)))
            case "ctd":
                blocks.append(ctd_descriptor(sequence_for(name)))

    descriptor = np.concatenate(blocks).astype(np.float32)
    expected = len(descriptor_names(sources))
    if len(descriptor) != expected:
        raise ValueError(f"expected {expected} values, built {len(descriptor)}")
    return descriptor


def describe_all(names: list[str], sources: tuple[str, ...] = ("curated",)) -> NDArray[np.float32]:
    """Descriptors for a list of enzymes, one row each."""
    logger.info("describing %d enzymes from %s", len(names), ", ".join(sources))
    return np.stack([enzyme_descriptor(name, sources) for name in names])


def descriptor_conditions(
    frame: pd.DataFrame,
    assays: tuple[str, ...],
    sources: tuple[str, ...] = ("curated",),
) -> NDArray[np.float32]:
    """Per-row conditioning block: the enzyme's pocket descriptor plus an assay one-hot.

    This is the ``descriptor`` conditioning input for a scalar model -- the pocket
    vector standing in for the target's one-hot, with the assay still one-hot because it
    is not an enzyme property. Rows are keyed by the ``target`` and ``assay`` columns.

    Raises:
        ValueError: On an assay outside ``assays`` -- the same contract as the one-hot
            encoders.
    """
    import numpy as np

    assay_of = {name: i for i, name in enumerate(assays)}
    unknown = set(frame["assay"]) - set(assay_of)
    if unknown:
        raise ValueError(f"unknown assay(s) {unknown}")
    vectors = {name: enzyme_descriptor(name, sources) for name in frame["target"].unique()}
    rows = []
    for target, assay in zip(frame["target"], frame["assay"], strict=True):
        one_hot = np.zeros(len(assays), dtype=np.float32)
        one_hot[assay_of[assay]] = 1.0
        rows.append(np.concatenate([vectors[target], one_hot]))
    return np.asarray(rows, dtype=np.float32)


def descriptor_condition_width(assays: tuple[str, ...], sources: tuple[str, ...]) -> int:
    """Width of the block :func:`descriptor_conditions` produces."""
    return len(source_names_width(sources)) + len(assays)


def source_names_width(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Descriptor column names for the given sources (for width and provenance)."""
    return descriptor_names(sources)
