```
import pandas as pd
import matplotlib.pyplot as plt

def dist_counts(df, sat_col, group_col, sat_min=0, sat_max=None, dropna=True):
    """
    Returns:
      counts: table of counts (rows = sat values, cols = groups)
      pct:    table of percentages normalized to 100% within each group (each column sums to 100)
    """
    tmp = df[[sat_col, group_col]].copy()
    if dropna:
        tmp = tmp.dropna(subset=[sat_col, group_col])

    if sat_max is None:
        sat_max = int(pd.to_numeric(tmp[sat_col], errors="coerce").max())

    sat_order = list(range(sat_min, sat_max + 1))

    counts = pd.crosstab(tmp[sat_col], tmp[group_col]).reindex(index=sat_order, fill_value=0)
    pct = counts.div(counts.sum(axis=0), axis=1).mul(100)

    return counts, pct


def plot_multibar(pct, sat_col="sat", group_col="group", title=None, figsize=(12, 5)):
    """
    pct should be the percentage table from dist_counts (rows=sat, cols=groups).
    Makes a grouped (multi-bar) plot where each group is normalized to 100%.
    """
    ax = pct.T.plot(kind="bar", figsize=figsize)  # multi-bar (not stacked)
    ax.set_ylim(0, 100)
    ax.set_xlabel(group_col)
    ax.set_ylabel(f"% within each {group_col}")
    ax.set_title(title or f"{sat_col} distribution by {group_col} (100% normalized)")
    ax.legend(title=sat_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

counts, pct = dist_counts(df, sat_col="sat", group_col="programmes", sat_min=0, sat_max=10)
plot_multibar(pct, sat_col="sat", group_col="programmes")



```
