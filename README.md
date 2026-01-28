```

import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_distribution_with_labels(
    counts_table,     # counts (rows=target, cols=groups)
    pct_table,        # pct (same shape)
    title="Grouped distribution",
    xlabel=None,
    ylabel="Percentage (%)",
    cmap_name="Set2",
    figsize=(12, 6),
    grid=True,
    legend_title=None,
    min_pct_label=3.0  # don't label tiny segments (<3%)
):
    # Plot stacked % (pct_table)
    ax = pct_table.plot(
        kind="bar",
        stacked=True,
        figsize=figsize,
        colormap=cmap_name,
        edgecolor="black",
        linewidth=0.3
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel if xlabel else (pct_table.index.name or ""))
    plt.ylabel(ylabel)

    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.35)

    # Add labels inside each stacked segment: "count\nxx.xx%"
    # IMPORTANT: ax.containers order matches columns order in pct_table
    for j, container in enumerate(ax.containers):
        col_name = pct_table.columns[j]

        for i, rect in enumerate(container):
            pct_val = float(pct_table.iloc[i, j])
            cnt_val = int(counts_table.iloc[i, j])

            # skip very small segments
            if pct_val < min_pct_label or cnt_val == 0:
                continue

            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2

            ax.text(
                x, y,
                f"{cnt_val}\n{pct_val:.1f}%",
                ha="center", va="center",
                fontsize=8, color="black"
            )

    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


```

```
counts, pct, combined = grouped_distribution_table(
    df,
    target_col="evaluate_note",
    group_col="decision_ai",
    normalize="group"
)

plot_grouped_distribution_with_labels(
    counts_table=counts,
    pct_table=pct,
    title="Distribution of evaluate_note by decision_ai (count + %)",
    xlabel="Evaluation note (0–10)",
    legend_title="decision_ai",
    cmap_name="Set2",
    figsize=(12, 6),
    min_pct_label=2.0
)




```
