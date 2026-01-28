```
import pandas as pd
import numpy as np

def grouped_distribution_table(
    df,
    target_col,         # e.g. "evaluate_note"
    group_col,          # e.g. "decision_ai"
    normalize="group",  # "group" => % inside each group, "all" => % of total dataset
    top_n=None,         # keep top categories of target_col (optional)
    other_label="Other",
    include_na=False
):
    d = df[[target_col, group_col]].copy()

    if not include_na:
        d = d.dropna(subset=[target_col, group_col])

    # counts: rows=target, cols=group
    counts = (
        d.groupby([target_col, group_col])
         .size()
         .unstack(fill_value=0)
    )

    # optional: keep top_n target values
    if top_n is not None and len(counts) > top_n:
        top_idx = counts.sum(axis=1).sort_values(ascending=False).head(top_n).index
        other_row = counts.drop(index=top_idx).sum(axis=0)
        counts = counts.loc[top_idx]
        counts.loc[other_label] = other_row

    # percentages
    if normalize == "group":
        pct = (counts.div(counts.sum(axis=0), axis=1) * 100).round(2)  # % within each group column
    elif normalize == "all":
        pct = (counts / counts.to_numpy().sum() * 100).round(2)        # % of whole dataset
    else:
        raise ValueError("normalize must be 'group' or 'all'")

    # combined table "count (xx.xx%)"
    combined = counts.astype(int).astype(str) + " (" + pct.astype(str) + "%)"

    return counts, pct, combined


```


```
import matplotlib.pyplot as plt

def plot_grouped_distribution(
    pct_table,          # table from grouped_distribution_table (pct)
    title="Grouped distribution",
    xlabel=None,
    ylabel="Percentage (%)",
    cmap_name="Set2",
    grid=True,
    figsize=(12, 6),
    legend_title=None
):
    # pct_table: rows=target values, cols=groups
    ax = pct_table.plot(kind="bar", stacked=True, figsize=figsize, colormap=cmap_name, edgecolor="black", linewidth=0.3)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel if xlabel else pct_table.index.name if pct_table.index.name else "")
    plt.ylabel(ylabel)

    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.35)

    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


```

```
counts, pct, combined = grouped_distribution_table(
    df,
    target_col="evaluate_note",
    group_col="decision_ai",
    normalize="group"   # % inside decision_ai groups
)


from tabulate import tabulate

table = combined.reset_index().rename(columns={"evaluate_note": "Evaluation note"})
print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))


plot_grouped_distribution(
    pct,
    title="Distribution of evaluate_note by decision_ai",
    xlabel="Evaluation note (0–10)",
    legend_title="decision_ai",
    cmap_name="Set2"
)


# evaluate_note by dossier_complet
counts, pct, combined = grouped_distribution_table(df, "evaluate_note", "dossier_complet")
plot_grouped_distribution(pct, title="evaluate_note by dossier_complet", legend_title="dossier_complet")

# evaluate_note by PARCOURS_FINAL (many categories -> maybe limit groups first or limit target)
counts, pct, combined = grouped_distribution_table(df, "evaluate_note", "PARCOURS_FINAL")
plot_grouped_distribution(pct, title="evaluate_note by PARCOURS_FINAL", legend_title="PARCOURS_FINAL")


```
