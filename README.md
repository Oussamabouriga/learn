```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_satisfaction_by_age_bins(
    df: pd.DataFrame,
    age_col: str = "age",
    sat_col: str = "satisfaction",
    breaks=(18, 30, 35, 60),   # your list
    include_outside=False,     # True => adds (-inf, first] and [last, +inf)
    min_count=1,               # drop bins with less than this many people
    size_scale=30              # marker size scaling
):
    breaks = sorted(list(breaks))

    # Optional: include outside ranges
    if include_outside:
        bins = [-np.inf] + breaks + [np.inf]
    else:
        bins = breaks

    # Create intervals
    d = df[[age_col, sat_col]].dropna().copy()

    # Keep only ages within [first, last) if not including outside
    if not include_outside:
        d = d[(d[age_col] >= bins[0]) & (d[age_col] < bins[-1])]

    d["age_bin"] = pd.cut(
        d[age_col],
        bins=bins,
        right=False,          # [a, b)
        include_lowest=True
    )

    # Aggregate: mean satisfaction + count
    g = (d.groupby("age_bin", observed=True)
           .agg(mean_sat=(sat_col, "mean"),
                n=("age_bin", "size"))
           .reset_index())

    # Filter small bins
    g = g[g["n"] >= min_count].copy()
    if g.empty:
        raise ValueError("No bins left after filtering. Check your breaks / min_count / data range.")

    # X positions: midpoint of each interval
    intervals = g["age_bin"].astype("category").cat.categories
    # Map each category interval to its midpoint
    mid_map = {iv: (iv.left + iv.right) / 2 for iv in intervals if np.isfinite(iv.left) and np.isfinite(iv.right)}
    # For -inf/+inf bins (if include_outside=True), place them near edges
    if include_outside:
        for iv in intervals:
            if not np.isfinite(iv.left) and np.isfinite(iv.right):
                mid_map[iv] = iv.right - 1
            if np.isfinite(iv.left) and not np.isfinite(iv.right):
                mid_map[iv] = iv.left + 1

    g["x_mid"] = g["age_bin"].map(mid_map)
    g = g.sort_values("x_mid")

    # Marker sizes based on volume
    sizes = (g["n"] / g["n"].max()) * size_scale * 10

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(g["x_mid"], g["mean_sat"], marker="o", linewidth=2, label="Mean satisfaction")
    plt.scatter(g["x_mid"], g["mean_sat"], s=sizes)

    # Labels on points (mean + n)
    for _, r in g.iterrows():
        plt.annotate(f"{r['mean_sat']:.2f} (n={int(r['n'])})",
                     (r["x_mid"], r["mean_sat"]),
                     textcoords="offset points", xytext=(0, 8), ha="center")

    # X tick labels as intervals
    plt.xticks(g["x_mid"], g["age_bin"].astype(str), rotation=30, ha="right")

    plt.ylim(0, 10)
    plt.xlabel("Age intervals (custom breaks)")
    plt.ylabel("Satisfaction (mean)")
    plt.title("Mean satisfaction by custom age intervals (marker size = volume)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return g  # returns the aggregated table (mean + n) if you want it

# Example:
# l = [18, 30, 35, 60]
# plot_satisfaction_by_age_bins(df, age_col="age", sat_col="note", breaks=l)


plot_satisfaction_by_age_bins(df, breaks=[18, 25, 40, 60, 80])


plot_satisfaction_by_age_bins(df, breaks=[18, 30, 35, 60], include_outside=True)



```
