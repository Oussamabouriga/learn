```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(
    df,
    col,
    title=None,
    top_n=15,
    include_na=False,
    as_percent=False,
    sort="auto",           # "auto" | "index" | "count"
    other_label="Other",
    cmap_name="tab20",
    grid=True
):
    s = df[col]

    # value counts
    vc = s.value_counts(dropna=not include_na)

    # sorting
    if sort == "auto":
        is_numeric = pd.api.types.is_numeric_dtype(s)
        if is_numeric and vc.index.nunique() <= 30:
            vc = vc.sort_index()
        else:
            vc = vc.sort_values(ascending=False)
    elif sort == "index":
        vc = vc.sort_index()
    elif sort == "count":
        vc = vc.sort_values(ascending=False)

    # top N + other
    if top_n is not None and len(vc) > top_n:
        top = vc.iloc[:top_n]
        other = vc.iloc[top_n:].sum()
        vc = pd.concat([top, pd.Series({other_label: other})])

    labels = vc.index.astype(str)
    counts = vc.values
    total = counts.sum()
    pct = (counts / total * 100) if total > 0 else np.zeros_like(counts, dtype=float)

    y = pct if as_percent else counts
    ylabel = "Percentage (%)" if as_percent else "Count"

    # --- Adaptive sizing ---
    n = len(vc)
    max_label_len = max((len(x) for x in labels), default=1)

    # If many categories OR long labels => horizontal plot
    horizontal = (n >= 8) or (max_label_len >= 12)

    if horizontal:
        width = 10
        height = max(4, min(0.5 * n + 1.5, 14))   # grows with number of categories
    else:
        width = max(8, min(0.6 * n + 5, 14))
        height = 5

    plt.figure(figsize=(width, height))

    # Color palette
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # --- Plot ---
    if horizontal:
        bars = plt.barh(labels, y, color=colors, edgecolor="black", linewidth=0.3)
        plt.xlabel(ylabel)
        plt.ylabel(col)

        # Grid
        if grid:
            plt.grid(axis="x", linestyle="--", alpha=0.35)

        # Labels on bars
        x_pad = 0.01 * (max(y) if len(y) else 1)
        for i, b in enumerate(bars):
            val = b.get_width()
            txt = f"{counts[i]} ({pct[i]:.2f}%)"
            plt.text(val + x_pad, b.get_y() + b.get_height()/2, txt,
                     va="center", fontsize=9)

    else:
        bars = plt.bar(labels, y, color=colors, edgecolor="black", linewidth=0.3)
        plt.ylabel(ylabel)
        plt.xlabel(col)

        # Grid
        if grid:
            plt.grid(axis="y", linestyle="--", alpha=0.35)

        # Rotate x labels only if needed
        if max_label_len >= 6:
            plt.xticks(rotation=30, ha="right")

        # Labels on bars
        y_pad = 0.01 * (max(y) if len(y) else 1)
        for i, b in enumerate(bars):
            val = b.get_height()
            txt = f"{counts[i]}\n({pct[i]:.2f}%)"
            plt.text(b.get_x() + b.get_width()/2, val + y_pad, txt,
                     ha="center", va="bottom", fontsize=9)

    plot_title = title if title else f"Distribution of {col}"
    plt.title(plot_title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


```
```
plot_distribution(df, "evaluate_note", title="Distribution: evaluate_note (0–10)", sort="index", top_n=None, cmap_name="Set2")
plot_distribution(df, "PARCOURS_FINAL", title="Distribution: PARCOURS_FINAL", top_n=10, sort="count", cmap_name="tab20")
plot_distribution(df, "decision_ai", title="Distribution: Decision AI", sort="index", top_n=None, cmap_name="Set2")

```

```
plot_distribution(df, "evaluate_note", title="Distribution: evaluate_note (0–10)", sort="index", top_n=None)
plot_distribution(df, "decision_ai", title="Distribution: Decision AI", sort="index", top_n=None)


plot_distribution(df, "evaluate_note", cmap_name="Set2")
plot_distribution(df, "PARCOURS_FINAL", cmap_name="Dark2")
plot_distribution(df, "PARCOURS_FINAL", cmap_name="viridis")

cols_to_plot = ["evaluate_note", "decision_ai", "dossier_complet", "Nbr_ticket_pieces", "Nbr_ticket_information"]

for c in cols_to_plot:
    plot_distribution(df, c, top_n=12)

```
