```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_nps_by_age_delay(
    df: pd.DataFrame,
    age_col: str,
    delay_col: str,
    score_col: str,                 # evaluate_note column name
    age_breaks=(18, 30, 40, 50, 60, 70, 100),
    delay_breaks=(0, 5, 15, 30, 60, 120, 300, 600, 1500, 5000),
    age_range=(18, 100),
    delay_range=None,               # e.g. (0, 5000) or None
    promoters_min=9,                # NPS definition: promoters >= 9
    detractors_max=6,               # detractors <= 6
    min_n=1,                        # drop cells with fewer than this n
    label_bars=True,
    label_nps=True,
):
    d = df[[age_col, delay_col, score_col]].dropna().copy()

    # filter ranges
    d = d[(d[age_col] >= age_range[0]) & (d[age_col] <= age_range[1])]
    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    age_breaks = sorted(list(age_breaks))
    delay_breaks = sorted(list(delay_breaks))

    # bins
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)
    d["delay_bin"] = pd.cut(d[delay_col], bins=delay_breaks, right=False, include_lowest=True)

    # helper: compute NPS on a group of scores
    def nps(scores: pd.Series) -> float:
        scores = scores.dropna()
        if len(scores) == 0:
            return np.nan
        prom = (scores >= promoters_min).mean() * 100
        detr = (scores <= detractors_max).mean() * 100
        return prom - detr

    # aggregate per (age_bin, delay_bin)
    g = (d.groupby(["age_bin", "delay_bin"], observed=True)
           .agg(n=("age_bin", "size"),
                nps=(score_col, nps))
           .reset_index())

    g = g[g["n"] >= min_n].copy()
    if g.empty:
        raise ValueError("No data after filtering/bucketing. Check breaks/ranges/min_n.")

    # order delay bins for x axis
    delay_cats = g["delay_bin"].astype("category").cat.categories
    delay_labels = [str(iv) for iv in delay_cats]
    x = np.arange(len(delay_cats))

    # one subplot per age bin
    age_bins = g["age_bin"].astype("category").cat.categories
    n_rows = len(age_bins)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, max(3, 2.6 * n_rows)), sharex=True)

    if n_rows == 1:
        axes = [axes]

    for ax, age_iv in zip(axes, age_bins):
        sub = g[g["age_bin"] == age_iv].copy()

        # align to all delay bins (fill missing with 0 / nan)
        sub = sub.set_index("delay_bin").reindex(delay_cats)
        counts = sub["n"].fillna(0).to_numpy()
        nps_vals = sub["nps"].to_numpy()

        # bars (volume)
        bars = ax.bar(x, counts, alpha=0.9, label="Volume (n)")
        ax.set_ylabel(f"{age_iv}\ncount")

        # bar labels (n)
        if label_bars:
            for i, b in enumerate(bars):
                if counts[i] > 0:
                    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                            f"{int(counts[i])}", ha="center", va="bottom", fontsize=9)

        # NPS line on second axis
        ax2 = ax.twinx()
        ax2.plot(x, nps_vals, marker="o", linewidth=2, label="NPS")
        ax2.set_ylim(-100, 100)
        ax2.set_ylabel("NPS")

        # NPS labels
        if label_nps:
            for i, v in enumerate(nps_vals):
                if not np.isnan(v) and counts[i] > 0:
                    ax2.annotate(f"{v:.0f}",
                                 (x[i], v),
                                 textcoords="offset points",
                                 xytext=(0, 8),
                                 ha="center",
                                 fontsize=9)

        # nice grid
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(delay_labels, rotation=30, ha="right")
    axes[-1].set_xlabel(f"{delay_col} bins")

    fig.suptitle("Volume (bars) + NPS (line) by Age x Delay bins", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    return g  # returns the aggregated table (n + nps per cell)

# Example usage:
# age_bins = [18, 30, 35, 60, 100]
# delay_bins = [0, 30, 60, 120, 300, 600, 1500, 5000]
# plot_nps_by_age_delay(df,
#     age_col="age",
#     delay_col="delai_declaration",
#     score_col="evaluate_note",
#     age_breaks=age_bins,
#     delay_breaks=delay_bins,
#     age_range=(18, 100),
#     delay_range=(0, 5000)
# )

plot_nps_by_age_delay(
  df,
  age_col="AGE",
  delay_col="delay_minutes",
  score_col="evaluate_note",
  age_breaks=[18, 25, 35, 50, 65, 100],
  delay_breaks=[0, 10, 20, 45, 90, 180, 360, 1000, 5000]
)


```
