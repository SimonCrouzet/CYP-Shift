"""Assembling every measurement into one long table of observations.

A *structure-free* model is not a ligand-only one. It sees no 3D structure, but the
protein is still available to it as identity, and the experiment is available as
one_hot. Both belong in the input, because that is what lets a single model learn
shared chemistry in its lower layers while taking target-specific and
assay-specific behaviour from the one_hot block.

The alternative -- one model per isoform, as the first baseline here did -- transfers
nothing between them and cannot use a measurement made under a different condition at
all. That is expensive when the same compound was measured on four enzymes, under two
incubation conditions, and at a single concentration in a primary screen.

So every measurement becomes one row: which compound, which target, which assay, what
value, and how much the value should be trusted. Sources differ enormously in that
last respect, and the weights say so rather than pretending otherwise.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cypshift.config import ISOFORMS

logger = logging.getLogger(__name__)

#: The assay conditions a measurement can come from. These are genuinely different
#: experiments on the same protein, not noisy copies of one another -- grouped by the
#: detection chemistry that separates them at the bench, since a luminescent AC50 is not
#: on the same scale as a fluorogenic IC50. The first is the reference the calibration is
#: identified against (:data:`REFERENCE_ASSAY`); the challenge's own three come first, the
#: auxiliary detection-chemistry contexts after.
ASSAYS = (
    "direct_inhibition_drc",
    "direct_inhibition_screen",
    "tdi_drc",
    "fluorogenic_pic50",  # Octant Supersome, the challenge's own platform
    "luminescent_ac50",  # Veith / Tox21 P450-Glo bioluminescence
    "literature_ic50",  # ChEMBL / SAIR aggregated IC50 / Ki
    "reversible_pic50",  # Novartis internal reversible assay
)

#: The assay the shared potency is expressed in, and the one predictions are made at. Its
#: per-assay calibration is pinned to identity so the latent potency stays on this scale.
REFERENCE_ASSAY = "direct_inhibition_drc"

#: A qHTS max-response row (a 0-1 response magnitude, not a pIC50). Kept out of ``ASSAYS``
#: because it is routed to the graph model's dedicated response head rather than pooled
#: onto the potency scale; only that model sees it.
RESPONSE_ASSAY = "luminescent_max_response"

#: Proximity data_spaces, ordered from farthest to the scored endpoint to the endpoint itself.
#: This is *one* facet a training stage can select on -- how related the protein is: a
#: relative or noisy outside dataset (``distant_aux``), a scored enzyme's own outside rows
#: (``near_aux``), or the challenge's own measurements (``challenge``). The other facet is
#: the assay a row was measured in (:data:`ASSAYS`), the *type* of data; a stage can filter
#: on either or both, after the pattern of a proteochemometric corpus that separates target
#: relatedness from assay context. Both are stamped on every row when it is built.
DATA_SPACES = ("distant_aux", "near_aux", "challenge")

OBSERVATION_COLUMNS = (
    "Molecule_Name",
    "SMILES",
    "target",
    "assay",
    "value",
    "conf_low",
    "conf_high",
    "weight",
    "is_censored",
    # Proximity data_space of the row (see DATA_SPACES): challenge, or how far an auxiliary enzyme's
    # binding site is. The data-type facet is the ``assay`` column. A training stage
    # selects on either or both.
    "data_space",
    # Which dataset the row came from (see SOURCES). Same-chemistry sources pool into one
    # assay by default; source-stratified conditioning keeps them apart by this column.
    "source",
)

#: The datasets a row can originate from, for source-stratified conditioning. The challenge's
#: own direct and preincubated measurements share "challenge"; the single-point screen is its
#: own source; the rest are the auxiliary loaders.
SOURCES = ("challenge", "screen", "chembl", "sair", "octant", "veith", "tox21", "novartis")


@dataclass(frozen=True)
class SourceWeights:
    """How far each source is trusted, relative to a fitted dose-response curve.

    A screen pseudo-label carries the calibration residual on top of the assay's own
    noise, so it is down-weighted rather than excluded. The TDI assay_condition is a real curve
    but measures a different condition, so it informs the shared representation
    without being treated as a direct-inhibition measurement.
    """

    dose_response: float = 1.0
    screen_pseudo_label: float = 0.3
    tdi_assay_condition: float = 0.5


#: Module-level default, so the frozen instance is shared rather than rebuilt per call.
DEFAULT_WEIGHTS = SourceWeights()


def _from_wide(
    frame: pd.DataFrame,
    assay: str,
    value_template: str,
    weight: float,
    censored_template: str | None = None,
    source: str = "challenge",
) -> pd.DataFrame:
    """Melt one wide source into observation rows, one per compound and target."""
    rows = []
    for isoform in ISOFORMS:
        column = value_template.format(isoform=isoform)
        if column not in frame:
            continue
        present = frame[column].notna()
        if not present.any():
            continue
        block = pd.DataFrame(
            {
                "Molecule_Name": frame.loc[present, "Molecule_Name"].to_numpy(),
                "SMILES": frame.loc[present, "SMILES"].to_numpy(),
                "target": isoform,
                "assay": assay,
                "value": frame.loc[present, column].to_numpy(float),
                "weight": weight,
            }
        )
        for bound in ("conf_low", "conf_high"):
            bound_column = f"{column}_{bound}"
            block[bound] = (
                frame.loc[present, bound_column].to_numpy(float) if bound_column in frame else np.nan
            )
        flag = censored_template.format(isoform=isoform) if censored_template else None
        block["is_censored"] = (
            frame.loc[present, flag].to_numpy(bool) if flag and flag in frame else False
        )
        block["data_space"] = "challenge"  # the challenge's own data; its assay carries the type
        block["source"] = source
        rows.append(block)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=OBSERVATION_COLUMNS)


def build_observations(
    inhibition: pd.DataFrame,
    screen_labels: pd.DataFrame | None = None,
    tdi: pd.DataFrame | None = None,
    weights: SourceWeights = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Stack every available measurement into one observation table.

    Args:
        inhibition: The dose-response direct-inhibition training set.
        screen_labels: Calibrated pseudo-labels from
            :func:`cypshift.data.screen.pseudo_labels`, or None to omit them.
        tdi: The time-dependent inhibition assay_condition, or None to omit it.
        weights: Relative trust in each source.

    Returns:
        One row per measurement with the columns in :data:`OBSERVATION_COLUMNS`.
    """
    sources = [
        _from_wide(
            inhibition,
            "direct_inhibition_drc",
            "{isoform}_pIC50_direct_inhibition",
            weights.dose_response,
        )
    ]
    if screen_labels is not None:
        sources.append(
            _from_wide(
                screen_labels,
                "direct_inhibition_screen",
                "{isoform}_pIC50_direct_inhibition",
                weights.screen_pseudo_label,
                censored_template="{isoform}_is_censored",
                source="screen",
            )
        )
    if tdi is not None:
        sources.append(
            _from_wide(tdi, "tdi_drc", "{isoform}_pIC50_TDI_condition", weights.tdi_assay_condition)
        )

    observations = pd.concat(sources, ignore_index=True)[list(OBSERVATION_COLUMNS)]
    summary = observations.groupby(["assay", "target"]).size().unstack(fill_value=0)
    logger.info("assembled %d observations:\n%s", len(observations), summary.to_string())
    return observations


#: Scalar properties of each assay, so the encoding can describe the experiment rather
#: than merely index it. A model given these can learn "preincubation" as one direction
#: shared across every enzyme, instead of treating each (enzyme, condition) pair as an
#: unrelated column.
#:
#: Four definitional properties: whether the assay preincubates with NADPH, whether it
#: reads a dose-response curve or a single concentration, whether the readout is a coupled
#: luminescent one (Veith/Tox21 P450-Glo) rather than the reference's fluorogenic probe,
#: and whether the values are a multi-lab literature aggregate (ChEMBL/SAIR) rather than a
#: single platform. The last two describe the auxiliary contexts so ``metadata`` is not
#: blind to detection chemistry; the residual systematic offset each still carries is the
#: per-assay calibration's job, not this vector's.
#:
#: A concentration column was tried and removed: the screen value was measured
#: (``concentration_M`` is 4.95049505e-05 throughout that table) but the dose-response
#: assay_conditions carry no concentration column, so the -4.0 used for them was invented, and it
#: was
#: a deterministic function of ``dose_response`` anyway.
ASSAY_METADATA = {
    #                          preincubated  dose_response  luminescent  aggregate
    "direct_inhibition_drc": (0.0, 1.0, 0.0, 0.0),  # fluorogenic, single lab (reference)
    "tdi_drc": (1.0, 1.0, 0.0, 0.0),
    "direct_inhibition_screen": (0.0, 0.0, 0.0, 0.0),
    # Auxiliary contexts described by what separates them at the bench: a coupled
    # luminescent readout (Veith/Tox21) and multi-lab literature aggregation (ChEMBL/SAIR)
    # are the real distinctions; Octant and Novartis share the reference's fluorogenic
    # single-lab profile, and the per-assay calibration carries whatever offset remains.
    "fluorogenic_pic50": (0.0, 1.0, 0.0, 0.0),
    "luminescent_ac50": (0.0, 1.0, 1.0, 0.0),
    "literature_ic50": (0.0, 1.0, 0.0, 1.0),
    "reversible_pic50": (0.0, 1.0, 0.0, 0.0),
    # The graph's response head reads the same P450-Glo luminescent curve, so it shares the
    # luminescent dose-response profile; without an entry, metadata conditioning raised.
    RESPONSE_ASSAY: (0.0, 1.0, 1.0, 0.0),
}
N_METADATA_COLUMNS = 4


def encode_conditions(
    observations: pd.DataFrame,
    strategy: str = "factorised",
    targets: tuple[str, ...] = ISOFORMS,
    assays: tuple[str, ...] = ASSAYS,
    sources: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Encode which target and assay each observation came from.

    Args:
        observations: Rows carrying ``target`` and ``assay`` columns.
        strategy: ``factorised`` is a target one-hot beside an *assay one-hot* -- assay
            identity. ``metadata`` is the target one-hot beside *scalar assay
            properties* (preincubation, dose-response) and no assay one-hot, so the
            assay enters by what it is rather than which one it is. The two are distinct
            encodings: metadata's scalars would be redundant with an assay one-hot if
            both were present, so metadata replaces it rather than adding to it.
        targets: Target vocabulary.
        assays: Assay vocabulary.

    Returns:
        Shape ``(n_observations, n_columns)``.

    Raises:
        ValueError: On an unknown strategy, target or assay.
    """
    if strategy == "factorised":
        base = one_hot_encode(observations, targets, assays)
    elif strategy == "metadata":
        unknown_targets = set(observations["target"]) - set(targets)
        unknown_assays = set(observations["assay"]) - set(assays)
        if unknown_targets or unknown_assays:
            raise ValueError(f"unrecognised target {unknown_targets} or assay {unknown_assays}")
        target_block = np.array(
            [[row == t for t in targets] for row in observations["target"]], dtype=np.float32
        )
        scalars = np.array(
            [ASSAY_METADATA[assay] for assay in observations["assay"]], dtype=np.float32
        )
        base = np.hstack([target_block, scalars])
    else:
        raise ValueError(f"unknown conditioning strategy {strategy!r}")
    if sources is None:
        return base
    # Source-stratified conditioning: append a one-hot over which dataset the row came from,
    # so same-chemistry sources kept apart in assembly each get their own calibration.
    unknown_sources = set(observations["source"]) - set(sources)
    if unknown_sources:
        raise ValueError(f"unrecognised source {unknown_sources}")
    source_block = np.array(
        [[row == s for s in sources] for row in observations["source"]], dtype=np.float32
    )
    return np.hstack([base, source_block]).astype(np.float32)


def n_condition_columns(
    strategy: str = "factorised",
    targets: tuple[str, ...] = ISOFORMS,
    assays: tuple[str, ...] = ASSAYS,
    sources: tuple[str, ...] | None = None,
) -> int:
    """Width of the encoding a strategy produces, without building it."""
    extra = len(sources) if sources is not None else 0
    if strategy == "factorised":
        return len(targets) + len(assays) + extra
    if strategy == "metadata":
        return len(targets) + N_METADATA_COLUMNS + extra
    raise ValueError(f"unknown conditioning strategy {strategy!r}")


def n_conditions(targets: tuple[str, ...] = ISOFORMS, assays: tuple[str, ...] = ASSAYS) -> int:
    """Number of distinct (target, assay) pairs, i.e. the embedding vocabulary size."""
    return len(targets) * len(assays)


def condition_index(
    observations: pd.DataFrame,
    targets: tuple[str, ...] = ISOFORMS,
    assays: tuple[str, ...] = ASSAYS,
) -> np.ndarray:
    """Map each row to a single (target, assay) index, for an embedding lookup.

    The one-hot encoders spend a column per level; an embedding indexes a learned table
    instead, so it needs one integer per row. Index is ``target * n_assays + assay`` in
    the given vocabularies.

    Raises:
        ValueError: On an unknown target or assay -- the same contract as
            :func:`encode_conditions`.
    """
    target_of = {name: i for i, name in enumerate(targets)}
    assay_of = {name: i for i, name in enumerate(assays)}
    unknown_t = set(observations["target"]) - set(target_of)
    unknown_a = set(observations["assay"]) - set(assay_of)
    if unknown_t or unknown_a:
        raise ValueError(f"unknown target(s) {unknown_t} or assay(s) {unknown_a}")
    t = observations["target"].map(target_of).to_numpy()
    a = observations["assay"].map(assay_of).to_numpy()
    return (t * len(assays) + a).astype(np.int64)


def one_hot_encode(
    observations: pd.DataFrame,
    targets: tuple[str, ...] = ISOFORMS,
    assays: tuple[str, ...] = ASSAYS,
) -> np.ndarray:
    """Encode the target and assay a row was measured under, for conditioning on them.

    Conditioning is the point; one-hot is only the current implementation of it. A
    learned protein embedding, a FiLM-style modulation of the trunk, or cross-attention
    over a sequence representation would condition on the same information differently,
    and swapping this function is how that would be tried.

    One-hot is chosen here for two measured reasons rather than by default. With four
    targets an embedding can produce only four distinct vectors, so it degenerates into
    a lookup table carrying a homology prior with nothing to generalise across --
    every target seen at prediction time was also seen in training. And it costs
    width: dense columns against a sparse molecular block dilute the latter.
    """
    unknown_targets = set(observations["target"]) - set(targets)
    unknown_assays = set(observations["assay"]) - set(assays)
    if unknown_targets or unknown_assays:
        raise ValueError(f"unrecognised target {unknown_targets} or assay {unknown_assays}")
    target_block = np.array([[row == t for t in targets] for row in observations["target"]])
    assay_block = np.array([[row == a for a in assays] for row in observations["assay"]])
    return np.hstack([target_block, assay_block]).astype(np.float32)
