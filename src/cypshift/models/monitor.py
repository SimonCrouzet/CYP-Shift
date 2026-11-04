"""Per-epoch loss curves on a cross-validation fold's held-out compounds, and pruning against a
reference run's curves.

A graph fit given ``monitor`` rows evaluates them after every epoch and records the mean absolute
error of the predicted value (pIC50 units). That curve can stop a member whose error has stopped
improving (``model.graph.early_stopping_patience``, a fraction of the epochs) and can abandon a
configuration whose error at half its epochs is above a reference run's lowest error by more than a
tolerance (``model.graph.prune_reference``): :class:`PrunedError` is raised and the sweep records
the configuration as pruned.
"""

import json
import math
from pathlib import Path

CURVES_FILE = "curves.json"


class PrunedError(Exception):
    """A cross-validation fold's error at the pruning epoch exceeded the reference's limit."""


def prune_epoch(epochs: int) -> int:
    """The 1-based epoch at which a run of ``epochs`` is compared with the reference: its half."""
    return math.ceil(epochs / 2)


def patience_epochs(fraction: float, epochs: int) -> int:
    """Epochs without a lower error that stop a stage of ``epochs``: ``fraction`` of them, at
    least one; 0 when ``fraction`` is 0 (never stops)."""
    return math.ceil(fraction * epochs) if fraction > 0 else 0


def write_curves(directory: Path, curves: dict[int, list[dict]]) -> None:
    """Write the per-fold curves: one entry per member and stage, with its epochs and error."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CURVES_FILE).write_text(json.dumps(curves) + "\n")


def reference_error(reference: str, fold: int) -> float | None:
    """A reference run's lowest error for ``fold``: the mean over its first-stage curves (one per
    member) of each curve's minimum.

    ``reference`` is a run bundle directory or its curves file. None when the reference has no
    first-stage curve for the fold.
    """
    path = Path(reference)
    path = path / CURVES_FILE if path.is_dir() else path
    entries = json.loads(path.read_text()).get(str(fold), [])
    lowest = [min(entry["error"]) for entry in entries if entry["stage"] == 1 and entry["error"]]
    return sum(lowest) / len(lowest) if lowest else None
