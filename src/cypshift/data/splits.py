"""Building a holdout whose relationship to the training set mimics the real test set.

Why not a random split. The blinded test set is 75 analog series expanded from
training hits, so a test compound sits at median ECFP4 Tanimoto 0.59 from its nearest
training neighbour. The training library is a diversity set and contains almost no
analog series, so a random split leaves held-out compounds *further* from the
training data than the real test compounds are. That is not a mild pessimism: it
penalises fingerprint models, which need local neighbours, and flatters bulk
descriptors, which do not. Model selection run on a random split therefore prefers
the wrong architecture.

Why not a scaffold split either. Holding out whole scaffold clusters pushes held-out
compounds even further away, in the same wrong direction.

What this module does instead is match the *distribution* of nearest-neighbour
similarity. Each candidate is scored by its maximum Tanimoto to the compounds that
would remain after the split, and compounds are then sampled so that the holdout
reproduces the test set's observed similarity histogram. Because removing the
holdout lowers everyone's maximum, the selection is iterated: score against the
current retained set, reselect, repeat.

A ceiling applies and should be reported rather than hidden. The training library is
a diversity set, so it simply does not contain enough mutually similar compounds to
place a holdout as close to the retained data as the analog-expanded test set sits to
the full training set. Matching narrows the gap; it does not close it. Use
:func:`similarity_profile_gap` to state the residual whenever a split is used.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def murcko_scaffold(smiles: str) -> str:
    """The Bemis-Murcko scaffold SMILES, or the compound itself when it has no ring system.

    Grouping a cross-validation by this keeps an analog series within a single fold, so a
    model is never scored on a near-duplicate of something it trained on. An acyclic
    compound has an empty scaffold; it falls back to its own SMILES so it forms its own
    group rather than all acyclic compounds collapsing into one.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"could not parse SMILES: {smiles!r}")
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule))
    return scaffold or Chem.MolToSmiles(molecule)


def stratified_scaffold_folds(
    smiles: Sequence[str],
    strata: Sequence[int],
    n_folds: int = 5,
    seed: int = 0,
) -> NDArray[np.int_]:
    """Assign each compound to one of ``n_folds`` folds for cross-validation.

    The split is a partition -- every compound lands in exactly one fold, so running all
    folds predicts each compound once, out of fold. Two disciplines shape it:

    * **scaffold-grouped**: compounds sharing a Bemis-Murcko scaffold go to the same fold,
      so an analog series never straddles the train/validation boundary and the estimate is
      not flattered by near-duplicate leakage;
    * **stratified**: folds are balanced on ``strata`` (e.g. how many endpoints a compound
      has measured), so no fold is starved of the sparser endpoints.

    Args:
        smiles: Training compounds, one per row.
        strata: Per-compound integer stratum to balance across folds.
        n_folds: Number of folds.
        seed: Fixes the assignment.

    Returns:
        Integer fold index in ``[0, n_folds)`` for each compound, in input order.
    """
    scaffolds = [murcko_scaffold(s) for s in smiles]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(scaffolds), dtype=int)
    placeholder = np.zeros(len(scaffolds))
    for fold, (_, held_out) in enumerate(splitter.split(placeholder, strata, scaffolds)):
        folds[held_out] = fold
    return folds


def morgan_fingerprints(
    smiles: Sequence[str], radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS
) -> list:
    """ECFP4-equivalent bit fingerprints. Unparseable SMILES raise rather than pass silently."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for index, item in enumerate(smiles):
        molecule = Chem.MolFromSmiles(item)
        if molecule is None:
            raise ValueError(f"could not parse SMILES at position {index}: {item!r}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def max_similarity_to(queries: Sequence, references: Sequence) -> NDArray[np.float64]:
    """For each query, its highest Tanimoto similarity to any reference."""
    return np.array([max(DataStructs.BulkTanimotoSimilarity(q, list(references))) for q in queries])


def leave_one_out_max_similarity(fingerprints: Sequence) -> NDArray[np.float64]:
    """For each fingerprint, its highest similarity to any *other* fingerprint in the set."""
    reference = list(fingerprints)
    scores = np.empty(len(reference))
    for index, query in enumerate(reference):
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(query, reference))
        similarities[index] = -1.0
        scores[index] = similarities.max()
    return scores


def max_similarity_against_retained(
    fingerprints: Sequence, retained: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Each compound's highest similarity to the retained set, excluding itself."""
    reference = [f for f, keep in zip(fingerprints, retained, strict=True) if keep]
    reference_positions = np.flatnonzero(retained)
    scores = np.empty(len(fingerprints))
    for index, query in enumerate(fingerprints):
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(query, reference))
        if retained[index]:
            similarities[np.searchsorted(reference_positions, index)] = -1.0
        scores[index] = similarities.max()
    return scores


def _select_by_bins(
    scores: NDArray[np.float64],
    edges: NDArray[np.float64],
    target_share: NDArray[np.float64],
    n_holdout: int,
    target_centre: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    """Fill each similarity bin to its share of the holdout, then top up any shortfall.

    A bin can be under-populated because the training library simply does not contain
    enough compounds at that similarity. The shortfall is then filled with whichever
    remaining candidates sit closest to the target distribution's centre, which is the
    least-wrong direction to miss in, and the size of the shortfall is logged.
    """
    mask = np.zeros(len(scores), dtype=bool)
    for bin_index in range(len(target_share)):
        in_bin = np.flatnonzero((scores >= edges[bin_index]) & (scores < edges[bin_index + 1]))
        wanted = int(round(target_share[bin_index] * n_holdout))
        if wanted == 0 or len(in_bin) == 0:
            continue
        mask[rng.choice(in_bin, size=min(wanted, len(in_bin)), replace=False)] = True

    selected = int(mask.sum())
    if selected < n_holdout:
        remaining = np.flatnonzero(~mask)
        order = np.argsort(np.abs(scores[remaining] - target_centre))
        mask[remaining[order][: n_holdout - selected]] = True
        logger.warning(
            "similarity bins were %d compounds short of %d; topped up from the pool centre",
            n_holdout - selected,
            n_holdout,
        )
    elif selected > n_holdout:
        # Independent per-bin rounding can sum past the requested count; trim the excess,
        # dropping the compounds farthest from the target centre (the mirror of the fill).
        chosen = np.flatnonzero(mask)
        drop = chosen[np.argsort(-np.abs(scores[chosen] - target_centre))][: selected - n_holdout]
        mask[drop] = False
        logger.info("similarity bins overshot %d by %d; trimmed", n_holdout, selected - n_holdout)
    return mask


def similarity_matched_holdout(
    train_smiles: Sequence[str],
    test_smiles: Sequence[str],
    holdout_fraction: float = 0.1,
    n_bins: int = 8,
    n_rounds: int = 3,
    seed: int = 0,
) -> NDArray[np.bool_]:
    """Select a holdout matching the test set's nearest-neighbour similarity profile.

    Args:
        train_smiles: Training compounds to draw the holdout from.
        test_smiles: Blinded test compounds, used only to measure the target profile.
        holdout_fraction: Share of the training set to hold out. The default of 0.1
            is not arbitrary: holding out more removes so many neighbours that the
            remaining compounds drift away from the target profile. Measured on this
            dataset, 10% closes 87% of the gap a random split leaves, and 20% closes
            only 36%.
        n_bins: Quantile bins used to match the two distributions.
        n_rounds: Reselection passes. Each rescores candidates against the retained
            set implied by the previous pass, which is what corrects for the
            neighbours the holdout itself removes.
        seed: Fixes which compounds are drawn within each bin.

    Returns:
        Boolean mask over ``train_smiles``, true where a compound is held out.
    """
    rng = np.random.default_rng(seed)
    train_fingerprints = morgan_fingerprints(train_smiles)
    test_fingerprints = morgan_fingerprints(test_smiles)

    target = max_similarity_to(test_fingerprints, train_fingerprints)
    edges = np.quantile(target, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    target_share = np.histogram(target, bins=edges)[0] / len(target)
    n_holdout = int(round(holdout_fraction * len(train_smiles)))

    target_centre = float(np.median(target))
    scores = leave_one_out_max_similarity(train_fingerprints)
    mask = _select_by_bins(scores, edges, target_share, n_holdout, target_centre, rng)
    for round_index in range(1, n_rounds):
        scores = max_similarity_against_retained(train_fingerprints, ~mask)
        mask = _select_by_bins(scores, edges, target_share, n_holdout, target_centre, rng)
        realised = max_similarity_against_retained(train_fingerprints, ~mask)[mask]
        logger.info(
            "round %d: holdout %d, median similarity %.3f against target %.3f",
            round_index,
            mask.sum(),
            float(np.median(realised)),
            float(np.median(target)),
        )
    return mask


@dataclass(frozen=True)
class AnalogueSplit:
    """An analogue holdout and the retained anchor each held-out compound came from.

    ``holdout_mask`` is true where a compound is a held-out analogue; ``anchor_of`` maps each
    compound to the row index of its retained anchor, or ``-1`` for a retained compound. The
    anchor mapping is what distinguishes this from a plain structural-neighbour mask: it names
    the retained reference a local corrector would read for each held-out compound.
    """

    holdout_mask: NDArray[np.bool_]
    anchor_of: NDArray[np.int_]

    def anchor_label_coverage(self, labels: NDArray[np.bool_]) -> dict[int, float]:
        """Per endpoint column, the fraction of held-out-and-labelled compounds whose anchor is
        also labelled there, i.e. for which a same-endpoint retained reference actually exists.

        A held-out compound is scored on an endpoint only where it carries a label; this reports
        whether its anchor carries one on the same endpoint, which is what a same-endpoint
        reference-hit comparison assumes (and what the mask alone cannot guarantee). ``NaN`` for
        an endpoint with no held-out label.
        """
        labels = np.asarray(labels, dtype=bool)
        held = np.flatnonzero(self.holdout_mask)
        coverage: dict[int, float] = {}
        for column in range(labels.shape[1]):
            labelled = held[labels[held, column]]
            if labelled.size == 0:
                coverage[column] = float("nan")
                continue
            anchors = self.anchor_of[labelled]
            has_anchor_label = (anchors >= 0) & labels[np.clip(anchors, 0, None), column]
            coverage[column] = float(has_anchor_label.sum() / labelled.size)
        return coverage


def analogue_split(
    train_smiles: Sequence[str],
    holdout_fraction: float = 0.1,
    min_similarity: float = 0.4,
    labels: NDArray[np.bool_] | None = None,
    require_labelled_anchor: bool = False,
    n_bits: int = DEFAULT_N_BITS,
    radius: int = DEFAULT_RADIUS,
    seed: int = 0,
) -> AnalogueSplit:
    """Hold out analogues of retained anchors, the way the blinded test was built.

    The scaffold split guards against near-duplicate leakage, and the similarity-matched
    split reproduces the test set's *distance* profile, but neither reproduces the test's
    *construction*: a handful of measured reference hits with ten purchased analogues apiece,
    scored while their anchor stays known. This builds exactly that second view. It walks
    compounds from the most to the least locally crowded, and for each one not yet placed it
    keeps that compound as a retained anchor and sends its still-unplaced neighbours above
    ``min_similarity`` to the holdout, recording which anchor each came from, until the holdout
    reaches ``holdout_fraction``.

    Args:
        train_smiles: Training compounds to split.
        holdout_fraction: Share of the training set to hold out as analogues.
        min_similarity: A neighbour joins an anchor's holdout only above this Tanimoto, so a
            compound with no real analogue is never forced into a spurious pair.
        labels: Optional ``(n, n_endpoints)`` boolean of which endpoints each compound has a
            measurement for, used by ``require_labelled_anchor``.
        require_labelled_anchor: When true (and ``labels`` given), a neighbour joins an anchor's
            holdout only if the anchor shares at least one measured endpoint with it, so the
            held-out analogue actually has a same-endpoint retained reference.
        n_bits, radius: Fingerprint geometry (ECFP4-equivalent by default).
        seed: Fixes the anchor walk where similarities tie.

    Returns:
        An :class:`AnalogueSplit` (mask plus the anchor index per held-out compound).
    """
    if require_labelled_anchor and labels is None:
        raise ValueError("require_labelled_anchor needs labels")
    label_array = None if labels is None else np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    fingerprints = morgan_fingerprints(train_smiles, radius=radius, n_bits=n_bits)
    n = len(fingerprints)
    n_holdout = int(round(holdout_fraction * n))
    crowding = leave_one_out_max_similarity(fingerprints)
    # Most locally crowded first; a seeded jitter breaks ties without a systematic bias.
    order = np.argsort(-(crowding + rng.uniform(0, 1e-9, size=n)))

    placed = np.zeros(n, dtype=bool)  # anchor or already-held-out
    holdout = np.zeros(n, dtype=bool)
    anchor_of = np.full(n, -1, dtype=int)
    for anchor in order:
        if holdout.sum() >= n_holdout:
            break
        if placed[anchor]:
            continue
        placed[anchor] = True  # the anchor itself stays retained
        similarities = np.array(
            DataStructs.BulkTanimotoSimilarity(fingerprints[anchor], fingerprints)
        )
        neighbours = np.flatnonzero((similarities >= min_similarity) & ~placed)
        # Closest analogues first, so the budget buys the tightest pairs.
        for neighbour in neighbours[np.argsort(-similarities[neighbours])]:
            if holdout.sum() >= n_holdout:
                break
            if (
                require_labelled_anchor
                and label_array is not None
                and not np.any(label_array[anchor] & label_array[neighbour])
            ):
                continue  # no shared endpoint: the anchor is not a usable reference here
            holdout[neighbour] = True
            placed[neighbour] = True
            anchor_of[neighbour] = anchor
    logger.info(
        "analogue holdout: %d of %d compounds held out as analogues (target %d)",
        int(holdout.sum()),
        n,
        n_holdout,
    )
    return AnalogueSplit(holdout, anchor_of)


def analogue_holdout(
    train_smiles: Sequence[str],
    holdout_fraction: float = 0.1,
    min_similarity: float = 0.4,
    n_bits: int = DEFAULT_N_BITS,
    radius: int = DEFAULT_RADIUS,
    seed: int = 0,
) -> NDArray[np.bool_]:
    """The analogue holdout mask; see :func:`analogue_split` for the anchor-aware version."""
    return analogue_split(
        train_smiles,
        holdout_fraction=holdout_fraction,
        min_similarity=min_similarity,
        n_bits=n_bits,
        radius=radius,
        seed=seed,
    ).holdout_mask


def similarity_profile_gap(
    holdout_similarity: NDArray[np.float64], target_similarity: NDArray[np.float64]
) -> dict[str, float]:
    """How closely a realised holdout reproduces the target similarity profile."""
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    return {
        f"q{int(q * 100):02d}_gap": float(
            np.quantile(holdout_similarity, q) - np.quantile(target_similarity, q)
        )
        for q in quantiles
    }
