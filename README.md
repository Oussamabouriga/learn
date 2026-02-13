```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_volume_nps(
    df,
    delay_col,
    score_col,
    delay_breaks,          # still in MINUTES
    delay_range=None,      # still in MINUTES (tuple: (min, max))
    promoters_min=9,
    detractors_max=6,
):
    # Work in minutes internally (same logic)
    d = df[[delay_col, score_col]].dropna().copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    delay_breaks = sorted(delay_breaks)

    # Create delay bins in minutes (same)
    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    # NPS function
    def compute_nps(scores):
        if len(scores) == 0:
            return np.nan
        promoters = (scores >= promoters_min).mean() * 100
        detractors = (scores <= detractors_max).mean() * 100
        return promoters - detractors

    # Aggregate per delay bin
    g = (
        d.groupby("delay_bin", observed=True)
         .agg(
            volume=(delay_col, "size"),
            nps=(score_col, compute_nps)
         )
         .reset_index()
    )

    # ---- X labels: show bins in DAYS (but bins were built in minutes) ----
    def interval_minutes_to_days_label(iv):
        # iv is a pandas Interval (minutes)
        if pd.isna(iv):
            return "NaN"
        left_d = iv.left / 60 / 24
        right_d = iv.right / 60 / 24
        # nicer formatting: if huge, keep 2 decimals; else 2 decimals anyway
        return f"[{left_d:.2f}, {right_d:.2f}) days"

    x = np.arange(len(g))
    x_labels = g["delay_bin"].apply(interval_minutes_to_days_label).tolist()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars = volume
    bars = ax.bar(x, g["volume"], alpha=0.4)
    ax.set_ylabel("Volume")

    # Volume labels
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(g['volume'].iloc[i])}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # NPS line
    ax2 = ax.twinx()
    ax2.plot(x, g["nps"], marker="o", linewidth=2)
    ax2.set_ylabel("NPS")
    ax2.set_ylim(-100, 100)

    # NPS labels
    for i, val in enumerate(g["nps"]):
        if not np.isnan(val):
            ax2.annotate(
                f"{val:.0f}",
                (x[i], val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel(f"{delay_col} (shown in days, bins computed in minutes)")

    plt.title(f"NPS et Volume par {delay_col}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return g

```
