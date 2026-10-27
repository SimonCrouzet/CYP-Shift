"""Plot exported learning curves: training loss and held-out error per epoch, averaged over folds.

    python scripts/plot_learning_curves.py notes/learning_curves_capacity.csv \\
        --labels focus_both_budget_1217e0=base focus_graph_a_0738bf=A \\
        --variants focus_epochs_100_62e10e=focus_both_budget_1217e0 \\
        --metrics notes/cell_metrics.csv --output notes/learning_curves_capacity.png

The CSV is the output of ``export_learning_curves.py``. Each cell is one line (the mean over folds)
with an optional band from the lowest to the highest fold; the dot marks the epoch of the lowest
mean held-out error. The two panels have different units (the training loss is the objective's
value, the held-out error is a mean absolute error in pIC50), so each has its own axis.

A cell listed in ``--variants NAME=PARENT`` is the same model as PARENT with other training
settings: it takes the parent's colour and is drawn dashed.

A table under the panels gives, per cell, the lowest mean held-out error with its epoch, the last
epoch's error and the last training loss (all read off the mean curves), and with ``--metrics`` the
output of ``export_cell_metrics.py`` (Pearson, MAE, R², ST-RAE of the final predictions).
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- after the backend is chosen
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
#: Categorical slots 1-4 of the reference palette, in order; a cell keeps its colour.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
DASH = (0, (4, 2.5))

#: (key, heading, format) of the table's numeric columns.
METRIC_COLUMNS = (
    ("Pearson", "Pearson", "{:.4f}"),
    ("MAE", "MAE", "{:.4f}"),
    ("R2", "R²", "{:.4f}"),
    ("ST-RAE", "ST-RAE", "{:.4f}"),
)
CURVE_COLUMNS = (
    ("lowest", "lowest\nheld-out MAE", "{:.4f}"),
    ("epoch", "at epoch", "{:d}"),
    ("last", "last-epoch\nMAE", "{:.4f}"),
    ("loss", "last train\nloss", "{:.4f}"),
)

# Figure layout, in inches from the top edge.
PANEL_TOP, PANEL_HEIGHT, AXIS_LABELS, LEGEND_ROW, TABLE_ROW = 1.4, 3.35, 0.65, 0.26, 0.3

Style = tuple[str, str | tuple]


def mean_and_band(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per epoch: the mean over folds and the lowest and highest fold."""
    grouped = table.dropna(subset=[column]).groupby("epoch")[column]
    return pd.DataFrame({"mean": grouped.mean(), "low": grouped.min(), "high": grouped.max()})


def curve_summary(table: pd.DataFrame, name: str) -> dict[str, float]:
    """Lowest mean held-out error and its epoch; last held-out error and training loss (means)."""
    cell = table[table["name"] == name]
    error = mean_and_band(cell, "fold_mae")["mean"]
    loss = mean_and_band(cell, "train_loss")["mean"]
    return {
        "lowest": float(error.min()),
        "epoch": int(error.idxmin()),
        "last": float(error.iloc[-1]),
        "loss": float(loss.iloc[-1]) if len(loss) else float("nan"),
    }


def assign_styles(names: list[str], variants: dict[str, str]) -> dict[str, Style]:
    """Colour and line style per cell: a variant shares its parent's colour and is dashed; the
    other cells take the palette slots in order."""
    parents = [name for name in names if name not in variants]
    if len(parents) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} cells without a parent (one palette slot each)")
    colour = dict(zip(parents, SERIES, strict=False))
    for name, parent in variants.items():
        if parent not in colour:
            raise ValueError(f"{name}: its parent {parent} is not a plotted cell")
        colour[name] = colour[parent]
    return {name: (colour[name], DASH if name in variants else "-") for name in names}


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
    styles: dict[str, Style],
    column: str,
    gap: float,
    bands: bool,
) -> None:
    ends = {}
    for name in names:
        colour, style = styles[name]
        curve = mean_and_band(table[table["name"] == name], column)
        if bands:
            axis.fill_between(
                curve.index, curve["low"], curve["high"], color=colour, alpha=0.10, lw=0
            )
        axis.plot(curve.index, curve["mean"], color=colour, ls=style, lw=2, solid_capstyle="round")
        ends[name] = float(curve["mean"].iloc[-1])
        if column == "fold_mae":
            best = curve["mean"].idxmin()
            axis.plot(
                [best], [curve["mean"].min()], "o", ms=8, color=colour, mec=SURFACE, mew=2, zorder=5
            )
    if len(names) > len(SERIES):
        return  # the legend and the table carry the identities; end labels would collide
    last = int(table["epoch"].max())
    for name, height in place_labels(ends, gap).items():
        axis.annotate(
            labels[name],
            xy=(last, ends[name]),
            xytext=(last * 1.07, height),
            color=INK,
            fontsize=9,
            va="center",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": styles[name][0],
                "lw": 1,
                "shrinkA": 0,
                "shrinkB": 2,
            },
        )


def draw_table(
    axis: Axes,
    names: list[str],
    labels: dict[str, str],
    styles: dict[str, Style],
    curves: dict[str, dict[str, float]],
    metrics: pd.DataFrame | None,
) -> None:
    """One row per cell: a swatch in the cell's colour and line style, its label, the
    final-prediction metrics and the curve summary."""
    groups = []
    if metrics is not None:
        groups.append(("Final out-of-fold predictions (mean of the 4 enzymes)", METRIC_COLUMNS))
    groups.append(("Learning curve (mean of the 5 folds)", CURVE_COLUMNS))
    columns = [column for _, group in groups for column in group]
    first, width = 0.34, (1 - 0.34) / len(columns)
    axis.set_xlim(0, 1)
    axis.set_ylim(len(names) + 2.2, 0)
    axis.axis("off")
    start = first
    for heading, group in groups:
        end = start + width * len(group)
        axis.text(start + 0.008, 0.45, heading, color=INK, fontsize=9, va="center")
        axis.plot([start + 0.008, end - 0.008], [0.95, 0.95], color=GRID, lw=1.2)
        start = end
    for index, (_, heading, _) in enumerate(columns):
        axis.text(
            first + width * (index + 1) - 0.01,
            1.7,
            heading,
            color=INK_SECONDARY,
            fontsize=8.5,
            ha="right",
            va="center",
            linespacing=1.1,
        )
    for row, name in enumerate(names):
        y = 2.2 + row + 0.5
        if row % 2 == 0:
            axis.axhspan(y - 0.5, y + 0.5, color=GRID, alpha=0.35, lw=0)
        colour, style = styles[name]
        axis.plot([0.003, 0.021], [y, y], color=colour, ls=style, lw=2.4, solid_capstyle="butt")
        axis.text(0.03, y, labels[name], color=INK, fontsize=9, va="center")
        found = metrics[metrics["name"] == name] if metrics is not None else None
        for index, (key, _, template) in enumerate(columns):
            if key in curves[name]:
                value = curves[name][key]
            elif found is not None and len(found):
                value = float(found.iloc[0][key])
            else:
                continue
            text = "n/a" if pd.isna(value) else template.format(value)
            axis.text(
                first + width * (index + 1) - 0.01,
                y,
                text,
                color=INK,
                fontsize=9,
                ha="right",
                va="center",
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
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help="NAME=PARENT: NAME is PARENT with other training settings (same colour, dashed)",
    )
    parser.add_argument("--bands", action="store_true", help="also draw the lowest-to-highest fold")
    parser.add_argument(
        "--metrics",
        type=Path,
        help="export_cell_metrics.py output; adds Pearson, MAE, R2 and ST-RAE to the table",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mae-limits", type=float, nargs=2, default=(0.52, 0.60))
    parser.add_argument("--title", default="Learning curves, mean of the 5 cross-validation folds")
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    pairs = [item.split("=", 1) for item in args.labels]
    names = [name for name, _ in pairs]
    short = {name: text.split("|")[0] for name, text in pairs}
    legend = {name: text.split("|")[-1] for name, text in pairs}
    variants = dict(item.split("=", 1) for item in args.variants)
    try:
        styles = assign_styles(names, variants)
    except ValueError as error:
        parser.error(str(error))

    legend_columns = min(len(names), 4 if len(names) <= 4 else 3)
    legend_rows = math.ceil(len(names) / legend_columns)
    top = PANEL_TOP + LEGEND_ROW * (legend_rows - 1)
    table_height = TABLE_ROW * (len(names) + 2.2)
    height = top + PANEL_HEIGHT + AXIS_LABELS + table_height + 0.15

    def fraction(inches_from_top: float) -> float:
        return 1 - inches_from_top / height

    figure = plt.figure(figsize=(12.5, height), facecolor=SURFACE)
    panels = figure.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.93,
        top=fraction(top),
        bottom=fraction(top + PANEL_HEIGHT),
        wspace=0.30,
    )
    left, right = figure.add_subplot(panels[0]), figure.add_subplot(panels[1])
    table_axis = figure.add_axes((0.07, 0.01, 0.86, table_height / height), facecolor=SURFACE)
    low, high = args.mae_limits
    draw(left, table, names, short, styles, "train_loss", 0.0035, args.bands)
    draw(right, table, names, short, styles, "fold_mae", (high - low) * 0.06, args.bands)
    right.set_ylim(low, high)
    for axis, title, label in (
        (left, "Training loss", "interval loss on the training rows"),
        (
            right,
            "Held-out error (first epochs above the axis)",
            "mean absolute error on the held-out fold (pIC50)",
        ),
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
    metrics = pd.read_csv(args.metrics) if args.metrics else None
    curves = {name: curve_summary(table, name) for name in names}
    draw_table(table_axis, names, legend, styles, curves, metrics)
    figure.text(0.07, fraction(0.27), args.title, color=INK, fontsize=13, va="top")
    note = "line = mean over folds, dot = lowest point of the mean held-out curve"
    if variants:
        note += "; dashed = same model, other training settings"
    figure.text(
        0.07,
        fraction(0.55),
        note,
        color=INK_SECONDARY,
        fontsize=9,
        va="top",
    )
    handles = [
        Line2D([0], [0], color=styles[name][0], ls=styles[name][1], lw=2, label=legend[name])
        for name in names
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.065, fraction(0.78)),
        ncol=legend_columns,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        handlelength=2.4,
        columnspacing=1.6,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
