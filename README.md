```
import pandas as pd
import matplotlib.pyplot as plt

def plot_100pct_distribution(df, x_col, group_col, x_order=None, group_order=None, dropna=True):
    # 1) keep only needed cols
    tmp = df[[x_col, group_col]].copy()
    if dropna:
        tmp = tmp.dropna(subset=[x_col, group_col])

    # 2) counts table: rows=x values, cols=groups
    counts = pd.crosstab(tmp[x_col], tmp[group_col])

    # optional ordering
    if x_order is not None:
        counts = counts.reindex(index=x_order, fill_value=0)
    if group_order is not None:
        counts = counts.reindex(columns=group_order, fill_value=0)

    # 3) convert to percentages (EACH GROUP COLUMN sums to 100)
    pct = counts.div(counts.sum(axis=0), axis=1).mul(100)

    # 4) plot as 100% stacked bar per group (better comparison)
    ax = pct.T.plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"% within each {group_col}")
    ax.set_xlabel(group_col)
    ax.legend(title=x_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return counts, pct


counts, pct = plot_100pct_distribution(
    df,
    x_col="sat",
    group_col="programmes",
    x_order=range(11)   # forces 0..10 even if some missing
)


counts, pct = plot_100pct_distribution(df, x_col="any_score", group_col="any_group")
```


```
import pandas as pd
import matplotlib.pyplot as plt

def plot_satisfaction_vs_group(df, sat_col, group_col, sat_order=None, group_order=None, dropna=True):
    tmp = df[[sat_col, group_col]].copy()
    if dropna:
        tmp = tmp.dropna(subset=[sat_col, group_col])

    # counts: rows = satisfaction values, cols = groups
    counts = pd.crosstab(tmp[sat_col], tmp[group_col])

    if sat_order is not None:
        counts = counts.reindex(index=sat_order, fill_value=0)
    if group_order is not None:
        counts = counts.reindex(columns=group_order, fill_value=0)

    # A) % within each group (each column sums to 100) => satisfaction distribution per group
    pct_within_group = counts.div(counts.sum(axis=0), axis=1).mul(100)

    ax = pct_within_group.T.plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_ylim(0, 100)
    ax.set_title(f"{sat_col} distribution within each {group_col} (100%)")
    ax.set_xlabel(group_col)
    ax.set_ylabel("% of people (within group)")
    ax.legend(title=sat_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # B) % within each satisfaction value (each row sums to 100) => group distribution per satisfaction
    pct_within_sat = counts.div(counts.sum(axis=1), axis=0).mul(100)

    ax = pct_within_sat.plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_ylim(0, 100)
    ax.set_title(f"{group_col} distribution for each {sat_col} value (100%)")
    ax.set_xlabel(sat_col)
    ax.set_ylabel("% of people (within satisfaction)")
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return counts, pct_within_group, pct_within_sat


counts, pct_group, pct_sat = plot_satisfaction_vs_group(
    df,
    sat_col="sat",
    group_col="programmes",
    sat_order=range(11)
)

sat_order=range(df["sat"].max() + 1)




```
