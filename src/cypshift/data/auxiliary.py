"""Choosing auxiliary enzymes by binding-site distance, and weighting their data by it.

Adding an auxiliary enzyme trades two things against each other. Its rows are real
measurements and enlarge a small training set, but they were measured on a different
protein, so whatever structure-activity relationship they carry is only approximately
the one we want. How approximate depends on how alike the binding sites are, which
:func:`cypshift.data.site.site_distance` measures -- so the same number can decide both
which enzymes to admit and how much to trust each one.

**Volume is the risk, not distance.** CYP2C19 alone contributes about 17,000 Veith rows
against roughly 6,000 dose-response labels across all four scored enzymes. Admitted at
equal weight, the auxiliaries decide the fit and the scored endpoints become a minority
of the loss. Downweighting by distance is what keeps an auxiliary contributing
proportionally to how much its site suggests it transfers.

The distance is a mean Z-scale difference per aligned binding-site position, so the
scale it lives on is worth keeping in mind when choosing a threshold:

===================================  ========
identical residue                    0.00
Leu to Ile, a conservative swap      0.62
mean over all 190 residue pairs      5.27
Asp to Arg, a charge reversal        5.67
===================================  ========

So 5.27 is what two unrelated sites score, and the default threshold of 1.5 is under a
third of that. Among human CYPs it admits seven enzymes; 3.0 admits seventeen, and the
additions are past halfway to the random baseline.

Three weighting schemes, because which is right is an empirical question:

``uniform``
    Every admitted enzyme counts fully. The baseline that says distance-weighting is
    doing nothing, and the one to beat.
``inverse``
    ``1 / (1 + distance / scale)``. Falls off gently; a distant enzyme keeps some say.
``exponential``
    ``exp(-distance / scale)``. Falls off sharply, so admitting a distant enzyme costs
    little. Closer to a kernel, and the more conservative choice.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cypshift.data.motifs import p450_score
from cypshift.data.site import site_distance

logger = logging.getLogger(__name__)

#: How an admitted enzyme's weight falls off with binding-site distance.
AUXILIARY_WEIGHTING = ("uniform", "inverse", "exponential")

#: Default admission threshold, in mean Z-scale units per aligned position. Under a
#: third of the 5.27 that two unrelated binding sites score.
DEFAULT_MAX_DISTANCE = 1.5

#: Distance at which ``inverse`` halves and ``exponential`` falls to 1/e. Roughly 1.6
#: conservative substitutions per position.
DEFAULT_SCALE = 1.0


@dataclass(frozen=True)
class Auxiliary:
    """An enzyme admitted for pretraining, and how far its site is from a scored one."""

    name: str
    nearest: str
    distance: float
    weight: float


def distance_weight(
    distance: float, scheme: str = "exponential", scale: float = DEFAULT_SCALE
) -> float:
    """Weight for one auxiliary enzyme's rows.

    Raises:
        ValueError: On an unknown scheme, or a scale that is not positive.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    match scheme:
        case "uniform":
            return 1.0
        case "inverse":
            return 1.0 / (1.0 + distance / scale)
        case "exponential":
            return float(np.exp(-distance / scale))
        case _:
            raise ValueError(f"unknown weighting {scheme!r}; expected one of {AUXILIARY_WEIGHTING}")


def site_distances(scored: tuple[str, ...], candidates: dict[str, str]) -> pd.DataFrame:
    """Binding-site distance from every candidate to every scored enzyme.

    Candidates that are not P450s, or whose site cannot be compared, are dropped with a
    log line. A scored enzyme appearing among the candidates is dropped too -- it is
    not auxiliary to itself.

    Args:
        scored: Gene symbols of the enzymes being predicted.
        candidates: Gene symbol to sequence.

    Returns:
        One row per usable candidate, a column per scored enzyme, plus ``nearest`` and
        ``distance`` naming the closest scored enzyme and how far it is.
    """
    from cypshift.data.sequences import sequence_for

    rows = []
    for name, sequence in candidates.items():
        if name in scored:
            continue
        if p450_score(sequence) < 0.7:
            logger.info("skipping %s: does not carry the P450 signature", name)
            continue
        try:
            measured = {t: site_distance(sequence_for(t), sequence) for t in scored}
        except (ValueError, KeyError) as error:
            logger.info("skipping %s: %s", name, error)
            continue
        nearest = min(scored, key=lambda t: measured[t])
        rows.append({"enzyme": name, **measured, "nearest": nearest, "distance": measured[nearest]})

    if not rows:
        return pd.DataFrame(columns=["enzyme", *scored, "nearest", "distance"]).set_index("enzyme")
    return pd.DataFrame(rows).sort_values("distance").set_index("enzyme")


def select_auxiliaries(
    distances: pd.DataFrame,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    weighting: str = "exponential",
    scale: float = DEFAULT_SCALE,
) -> list[Auxiliary]:
    """Admit every candidate within ``max_distance`` and weight it by how far it is.

    Returns:
        Admitted enzymes, closest first. Empty if nothing qualifies, which is the
        correct answer for CYP2D6 and worth surfacing rather than hiding.
    """
    admitted = [
        Auxiliary(
            name=str(name),
            nearest=str(row["nearest"]),
            distance=float(row["distance"]),
            weight=distance_weight(float(row["distance"]), weighting, scale),
        )
        for name, row in distances.iterrows()
        if float(row["distance"]) <= max_distance
    ]
    logger.info(
        "admitted %d of %d candidates within %.2f, weighting %s",
        len(admitted),
        len(distances),
        max_distance,
        weighting,
    )
    return admitted


def weight_observations(
    observations: pd.DataFrame,
    auxiliaries: list[Auxiliary],
    scored: tuple[str, ...],
    target_weight: float = 1.0,
    column: str = "target",
) -> pd.Series:
    """Per-row weight: ``target_weight`` for a scored enzyme, its own for an auxiliary.

    Rows from an enzyme that is neither get zero, so a table can carry more than was
    selected and the selection still decides what trains.

    Args:
        observations: Rows to weight.
        auxiliaries: What :func:`select_auxiliaries` admitted.
        scored: The enzymes being predicted. Required rather than inferred: taking
            "not an auxiliary" to mean "scored" would give an enzyme that *failed* the
            distance threshold full target weight, which is the opposite of the intent.
        target_weight: Weight for a scored enzyme's own rows.
        column: Column naming the enzyme.
    """
    lookup = {auxiliary.name: auxiliary.weight for auxiliary in auxiliaries}
    overlap = set(scored) & set(lookup)
    if overlap:
        raise ValueError(f"{sorted(overlap)} are both scored and auxiliary")
    weights = observations[column].map(
        lambda name: target_weight if name in set(scored) else lookup.get(name, 0.0)
    )
    return weights.astype(float)


def contribution(
    observations: pd.DataFrame, weights: pd.Series, column: str = "target"
) -> pd.DataFrame:
    """How much weighted mass each enzyme contributes, next to its raw row count.

    The check worth running before training: an auxiliary bringing 17,000 rows against
    6,000 target labels can still dominate the loss after downweighting, and a row
    count alone will not show it.
    """
    frame = pd.DataFrame({"enzyme": observations[column].to_numpy(), "weight": weights.to_numpy()})
    grouped = frame.groupby("enzyme")["weight"].agg(["count", "sum"])
    grouped["share"] = grouped["sum"] / grouped["sum"].sum()
    return grouped.sort_values("share", ascending=False)
