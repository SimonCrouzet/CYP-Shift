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
#: experiments on the same protein, not noisy copies of one another.
ASSAYS = ("direct_inhibition_drc", "direct_inhibition_screen", "tdi_drc")

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
)


@dataclass(frozen=True)
class SourceWeights:
    """How far each source is trusted, relative to a fitted dose-response curve.

    A screen pseudo-label carries the calibration residual on top of the assay's own
    noise, so it is down-weighted rather than excluded. The TDI arm is a real curve
    but measures a different condition, so it informs the shared representation
    without being treated as a direct-inhibition measurement.
    """

    dose_response: float = 1.0
    screen_pseudo_label: float = 0.3
    tdi_arm: float = 0.5


#: Module-level default, so the frozen instance is shared rather than rebuilt per call.
DEFAULT_WEIGHTS = SourceWeights()


def _from_wide(
    frame: pd.DataFrame,
    assay: str,
    value_template: str,
    weight: float,
    censored_template: str | None = None,
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
            source = f"{column}_{bound}"
            block[bound] = frame.loc[present, source].to_numpy(float) if source in frame else np.nan
        flag = censored_template.format(isoform=isoform) if censored_template else None
        block["is_censored"] = (
            frame.loc[present, flag].to_numpy(bool) if flag and flag in frame else False
        )
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
        tdi: The time-dependent inhibition arm, or None to omit it.
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
            )
        )
    if tdi is not None:
        sources.append(_from_wide(tdi, "tdi_drc", "{isoform}_pIC50_TDI_condition", weights.tdi_arm))

    observations = pd.concat(sources, ignore_index=True)[list(OBSERVATION_COLUMNS)]
    summary = observations.groupby(["assay", "target"]).size().unstack(fill_value=0)
    logger.info("assembled %d observations:\n%s", len(observations), summary.to_string())
    return observations


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
