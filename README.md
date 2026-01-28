```
import pandas as pd

def make_grouped_dist(df, x_col, group_col, group_order=None, x_order=None, normalize="x"):
    """
    normalize:
      - "x"   => percentage inside each x (each evaluate_note sums to 100%)
      - "all" => percentage over whole dataset
      - None  => no percentage
    """
    tmp = df[[x_col, group_col]].dropna()

    # counts matrix: rows=x, cols=group
    counts = tmp.groupby([x_col, group_col]).size().unstack(fill_value=0)

    if x_order is not None:
        counts = counts.reindex(index=x_order, fill_value=0)
    else:
        counts = counts.sort_index()

    if group_order is not None:
        counts = counts.reindex(columns=group_order, fill_value=0)
    else:
        counts = counts.sort_index(axis=1)

    if normalize == "x":
        pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(2)
    elif normalize == "all":
        pct = (counts / counts.values.sum() * 100).round(2)
    else:
        pct = None

    return counts, pct

import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_bars(counts, pct=None, title="", xlabel="", ylabel="Count",
                      group_labels=None, colors=None, show_pct=True, show_count=True,
                      figsize=(14,6), rotate_xticks=0):
    """
    counts: DataFrame (rows=x, cols=groups)
    pct:    DataFrame same shape as counts (optional)
    group_labels: dict mapping {original_col_value: "label"}
    colors: list of color names (one per group) e.g. ["steelblue","orange"]
    """

    x_vals = counts.index.tolist()
    groups = counts.columns.tolist()
    n_groups = len(groups)

    # labels
    if group_labels:
        legend_names = [group_labels.get(g, str(g)) for g in groups]
    else:
        legend_names = [str(g) for g in groups]

    # colors
    if colors is None:
        # default nice palette
        colors = ["steelblue", "seagreen", "orange", "slategray", "purple", "goldenrod"]
    colors = (colors * 10)[:n_groups]

    x = np.arange(len(x_vals))
    width = 0.8 / n_groups  # fill 80% of space

    fig, ax = plt.subplots(figsize=figsize)

    for i, g in enumerate(groups):
        y = counts[g].values
        bars = ax.bar(x + (i - (n_groups-1)/2)*width, y, width,
                      label=legend_names[i], color=colors[i], edgecolor="black", linewidth=0.3)

        # bar labels
        for j, b in enumerate(bars):
            c = int(y[j])
            if c == 0:
                continue

            parts = []
            if show_count:
                parts.append(str(c))
            if show_pct and pct is not None:
                parts.append(f"{pct[g].iloc[j]:.1f}%")
            txt = "\n".join(parts)

            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height(),
                    txt,
                    ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_vals, rotation=rotate_xticks)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


counts, pct = make_grouped_dist(
    df,
    x_col="evaluate_note",
    group_col="decision_ai",
    x_order=range(0, 11),
    group_order=[0, 1],
    normalize="x"
)

plot_grouped_bars(
    counts, pct,
    title="Evaluation note (0–10) by Decision AI — Grouped bars",
    xlabel="Evaluation note",
    ylabel="Count",
    group_labels={0: "Sans AI", 1: "Avec AI"},
    colors=["steelblue", "orange"],
    show_count=True,
    show_pct=True
)


counts, pct = make_grouped_dist(
    df,
    x_col="evaluate_note",
    group_col="dossier_complet",
    x_order=range(0, 11),
    group_order=[0, 1],
    normalize="x"
)

plot_grouped_bars(
    counts, pct,
    title="Evaluation note (0–10) by Dossier complet — Grouped bars",
    xlabel="Evaluation note",
    ylabel="Count",
    group_labels={0: "Non complet", 1: "Complet"},
    colors=["slategray", "seagreen"],
    show_count=True,
    show_pct=True
)


counts, pct = make_grouped_dist(
    df,
    x_col="evaluate_note",
    group_col="nombre_prestation_ko",
    x_order=range(0, 11),
    group_order=range(0, 6),
    normalize="x"
)

plot_grouped_bars(
    counts, pct,
    title="Evaluation note (0–10) by Nombre prestation KO (0–5) — Grouped bars",
    xlabel="Evaluation note",
    ylabel="Count",
    colors=["green", "lightgreen", "yellow", "orange", "red", "black"],
    show_count=True,
    show_pct=True,
    rotate_xticks=0
)




```
