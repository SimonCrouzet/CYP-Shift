"""Plot exported learning curves: training loss and held-out error per epoch, averaged over folds.

    python scripts/plot_learning_curves.py notes/learning_curves_capacity.csv \\
        --labels focus_both_budget_1217e0=base focus_graph_a_0738bf=A \\
        --output notes/learning_curves_capacity.png

The CSV is the output of ``export_learning_curves.py``. Each cell is one line (the mean over folds)
with a band from the lowest to the highest fold; the dot marks the epoch of the lowest mean
held-out error. The two panels have different units (the training loss is the objective's value,
the held-out error is a mean absolute error in pIC50), so each has its own axis.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- after the backend is chosen
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
#: Categorical slots 1-4 of the reference palette, in order; a cell keeps its colour.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def mean_and_band(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per epoch: the mean over folds and the lowest and highest fold."""
    grouped = table.dropna(subset=[column]).groupby("epoch")[column]
    return pd.DataFrame({"mean": grouped.mean(), "low": grouped.min(), "high": grouped.max()})


def place_labels(ends: dict[str, float], gap: float) -> dict[str, float]:
    """Label heights: the line ends, pushed apart to at least ``gap``, keeping their order."""
    names = sorted(ends, key=ends.get)
    heights = [ends[name] for name in names]
    for i in range(1, len(heights)):
        heights[i] = max(heights[i], heights[i - 1] + gap)
    shift = (sum(ends.values()) - sum(heights)) / len(heights)
    return {name: height + shift for name, height in zip(names, heights, strict=True)}


def draw(
    axis: Axes,
    table: pd.DataFrame,
    names: list[str],
    labels: dict[str, str],
    column: str,
    gap: float,
    bands: bool,
) -> None:
    ends = {}
    for name, colour in zip(names, SERIES, strict=False):
        curve = mean_and_band(table[table["name"] == name], column)
        if bands:
            axis.fill_between(
                curve.index, curve["low"], curve["high"], color=colour, alpha=0.10, lw=0
            )
        axis.plot(curve.index, curve["mean"], color=colour, lw=2, solid_capstyle="round")
        ends[name] = float(curve["mean"].iloc[-1])
        if column == "fold_mae":
            best = curve["mean"].idxmin()
            axis.plot(
                [best], [curve["mean"].min()], "o", ms=8, color=colour, mec=SURFACE, mew=2, zorder=5
            )
    last = int(table["epoch"].max())
    for name, height in place_labels(ends, gap).items():
        colour = SERIES[names.index(name)]
        axis.annotate(
            labels[name],
            xy=(last, ends[name]),
            xytext=(last + 2.2, height),
            color=INK,
            fontsize=9,
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": colour, "lw": 1, "shrinkA": 0, "shrinkB": 2},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="NAME=short or NAME=short|long (the long text is the legend's), in plotting order",
    )
    parser.add_argument("--bands", action="store_true", help="also draw the lowest-to-highest fold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mae-limits", type=float, nargs=2, default=(0.52, 0.60))
    parser.add_argument("--title", default="Learning curves, mean of the 5 cross-validation folds")
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    pairs = [item.split("=", 1) for item in args.labels]
    names = [name for name, _ in pairs]
    short = {name: text.split("|")[0] for name, text in pairs}
    legend = {name: text.split("|")[-1] for name, text in pairs}

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4), facecolor=SURFACE)
    figure.subplots_adjust(left=0.07, right=0.93, top=0.74, bottom=0.12, wspace=0.30)
    low, high = args.mae_limits
    draw(left, table, names, short, "train_loss", 0.0035, args.bands)
    draw(right, table, names, short, "fold_mae", (high - low) * 0.06, args.bands)
    right.set_ylim(low, high)
    for axis, title, label in (
        (left, "Training loss", "interval loss on the training rows"),
        (right, "Held-out error", "mean absolute error on the held-out fold (pIC50)"),
    ):
        axis.set_facecolor(SURFACE)
        axis.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
        axis.set_xlabel("epoch", color=INK_SECONDARY, fontsize=9)
        axis.set_ylabel(label, color=INK_SECONDARY, fontsize=9)
        axis.grid(True, color=GRID, lw=1, ls="-")
        axis.set_axisbelow(True)
        axis.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.set_xlim(1, int(table["epoch"].max()))
    figure.text(0.07, 0.95, args.title, color=INK, fontsize=13, va="top")
    figure.text(
        0.07,
        0.905,
        "line = mean over folds, dot = epoch of lowest mean held-out error; "
        "the first epochs of the held-out curve are above the axis limit",
        color=INK_SECONDARY,
        fontsize=9,
        va="top",
    )
    handles = [
        plt.Line2D([0], [0], color=colour, lw=2, label=legend[name])
        for name, colour in zip(names, SERIES, strict=False)
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.855),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        handlelength=1.6,
        columnspacing=1.6,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
