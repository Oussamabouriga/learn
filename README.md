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
    figsize=(10, 5),
    cmap_name="tab20"      # change to "Set2", "Dark2", "viridis", etc.
):
    """
    Pro distribution plot for any DataFrame column.
    - top_n: show top N categories; remaining aggregated into 'Other'
    - include_na: include NaN as a category
    - as_percent: plot percentages instead of counts
    - sort:
        * auto: numeric small categories -> sort by index; else sort by count
        * index: sort by category value
        * count: sort by frequency
    """

    s = df[col]

    # value counts
    vc = s.value_counts(dropna=not include_na)

    # Decide sorting
    if sort == "auto":
        is_numeric = pd.api.types.is_numeric_dtype(s)
        # if numeric and not too many unique values -> sort by index (0..10, 0..5, etc.)
        if is_numeric and vc.index.nunique() <= 30:
            vc = vc.sort_index()
        else:
            vc = vc.sort_values(ascending=False)
    elif sort == "index":
        vc = vc.sort_index()
    elif sort == "count":
        vc = vc.sort_values(ascending=False)

    # Top N + Other
    if top_n is not None and len(vc) > top_n:
        top = vc.iloc[:top_n]
        other = vc.iloc[top_n:].sum()
        vc = pd.concat([top, pd.Series({other_label: other})])

    # Convert index to string labels (for clean display)
    labels = vc.index.astype(str)
    values = vc.values

    # Percentages (for labels + optional plotting)
    total = values.sum()
    pct = (values / total * 100) if total > 0 else np.zeros_like(values)

    y = pct if as_percent else values
    ylabel = "Percentage (%)" if as_percent else "Count"

    # Choose orientation
    horizontal = len(vc) > 10  # many categories -> horizontal is clearer

    plt.figure(figsize=figsize)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(len(vc)-1, 1)) for i in range(len(vc))]

    if horizontal:
        bars = plt.barh(labels, y, color=colors)
        plt.xlabel(ylabel)
        plt.ylabel(col)
    else:
        bars = plt.bar(labels, y, color=colors)
        plt.ylabel(ylabel)
        plt.xlabel(col)
        plt.xticks(rotation=30, ha="right")

    plot_title = title if title else f"Distribution of {col}"
    plt.title(plot_title)

    # Add labels on bars: count + %
    for i, b in enumerate(bars):
        if horizontal:
            x = b.get_width()
            txt = f"{values[i]} ({pct[i]:.2f}%)"
            plt.text(x + (0.01 * max(y) if len(y) else 0), b.get_y() + b.get_height()/2,
                     txt, va="center", fontsize=9)
        else:
            h = b.get_height()
            txt = f"{values[i]}\n({pct[i]:.2f}%)"
            plt.text(b.get_x() + b.get_width()/2, h + (0.01 * max(y) if len(y) else 0),
                     txt, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

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
