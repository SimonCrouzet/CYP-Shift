"""Plot the held-out error of each enzyme's copy during its per-enzyme fine-tune.

    python scripts/plot_finetune_curves.py --rev questions3-m1 \\
        --cells focus_per_target_finetune_5cb805=base,30 epochs,lr 1e-4 \\
        --output notes/finetune_curves.png

A cell run with a per-enzyme fine-tune and ``CYP_MONITOR=1`` records, for each fold, one curve per
enzyme's copy after the pooled stages (``curves.json`` in the run bundle, stages ``len(stages) + 1``
onward, one per scored enzyme in the order CYP1A2, CYP2C9, CYP2D6, CYP3A4). Each panel is one
enzyme: the mean over folds of that copy's mean absolute error on the held-out compounds of the
enzyme, per epoch of the fine-tune. The error is not comparable across panels (each enzyme has its
own scale), only across the lines of one panel.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- after the backend is chosen
from plot_learning_curves import GRID, INK, INK_SECONDARY, SERIES, SURFACE  # noqa: E402

ENZYMES = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
BUNDLES = Path("data/run_bundles")


def branch_errors(curves: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Per enzyme, the mean over folds of its copy's held-out error per epoch, from a bundle's
    ``curves.json`` (fold -> entries)."""
    per_enzyme = {}
    for enzyme in ENZYMES:
        runs = [
            entry["error"]
            for entries in curves.values()
            for entry in entries
            if entry.get("target") == enzyme
        ]
        if runs:
            per_enzyme[enzyme] = np.mean(runs, axis=0)
    return per_enzyme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rev", required=True, help="the run's results namespace")
    parser.add_argument(
        "--cells",
        nargs="+",
        required=True,
        help="NAME=label: full recipe names and their legend text, in plotting order",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = [item.split("=", 1) for item in args.cells]
    if len(cells) > len(SERIES):
        parser.error(f"at most {len(SERIES)} cells (one palette slot each)")
    curves = {
        name: branch_errors(
            json.loads(
                (BUNDLES / args.rev / f"reg_{name}_seed{args.seed}" / "curves.json").read_text()
            )
        )
        for name, _ in cells
    }
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.8), facecolor=SURFACE)
    figure.subplots_adjust(left=0.06, right=0.98, top=0.68, bottom=0.13, wspace=0.28)
    for axis, enzyme in zip(axes, ENZYMES, strict=True):
        for (name, _), colour in zip(cells, SERIES, strict=False):
            if enzyme in curves[name]:
                error = curves[name][enzyme]
                axis.plot(range(1, len(error) + 1), error, color=colour, lw=2)
        axis.set_facecolor(SURFACE)
        axis.set_title(enzyme, loc="left", color=INK, fontsize=11, pad=8)
        axis.set_xlabel("epoch of the fine-tune", color=INK_SECONDARY, fontsize=9)
        axis.grid(True, color=GRID, lw=1)
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel("held-out MAE of the enzyme (pIC50)", color=INK_SECONDARY, fontsize=9)
    figure.text(
        0.06,
        0.96,
        "Per-enzyme fine-tune: held-out error of each enzyme's copy, mean of the 5 folds",
        color=INK,
        fontsize=13,
        va="top",
    )
    figure.text(
        0.06,
        0.905,
        "each panel has its own scale; compare the lines within a panel",
        color=INK_SECONDARY,
        fontsize=9,
        va="top",
    )
    handles = [
        plt.Line2D([0], [0], color=colour, lw=2, label=label)
        for (_, label), colour in zip(cells, SERIES, strict=False)
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.855),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
