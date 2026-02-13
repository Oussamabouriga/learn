```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_volume_nps(
    df,
    delay_col,
    score_col,
    delay_breaks,              # still in MINUTES (binning uses minutes)
    delay_range=None,          # (min_minutes, max_minutes) optional
    promoters_min=9,
    detractors_max=6,
    figsize=(12, 6),
    title=None
):
    """
    Bins delay_col using delay_breaks in MINUTES, computes:
      - Volume (count) per bin
      - NPS per bin (promoters% - detractors%)
    Plot:
      - Bars: volume
      - Line: NPS
    X-axis labels are shown in DAYS/HOURS (converted from minutes) ONLY for display.
    """

    # Keep only needed cols + dropna
    d = df[[delay_col, score_col]].dropna().copy()

    # Optional delay filter (still minutes)
    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    # Ensure breaks sorted
    delay_breaks = sorted(delay_breaks)

    # Create bins in minutes (binning logic stays minutes)
    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    # NPS function
    def compute_nps(scores: pd.Series) -> float:
        if len(scores) == 0:
            return np.nan
        promoters = (scores >= promoters_min).mean() * 100
        detractors = (scores <= detractors_max).mean() * 100
        return promoters - detractors

    # Aggregate per bin
    g = (
        d.groupby("delay_bin", observed=True)
         .agg(
             volume=(delay_col, "size"),
             nps=(score_col, compute_nps),
         )
         .reset_index()
    )

    # ---- Display labels in days/hours (ONLY formatting) ----
    def _format_days_hours(days_float: float) -> str:
        # days_float can be 0.x, etc.
        total_hours = int(round(days_float * 24))
        days = total_hours // 24
        hours = total_hours % 24

        if hours == 0:
            return f"{days}"
        if days == 0:
            return f"{hours}h"
        return f"{days}d {hours}h"

    def _interval_minutes_to_label(iv: pd.Interval) -> str:
        # iv.left/right are minutes
        left_days = iv.left / (60 * 24)
        right_days = iv.right / (60 * 24)

        left_lbl = _format_days_hours(left_days)
        right_lbl = _format_days_hours(right_days)

        # show unit nicely: if pure integer days -> "3" means 3 days
        # but keep 'd' when mixed with hours or hours-only
        def add_unit(lbl: str) -> str:
            if "d" in lbl or "h" in lbl:
                return lbl
            return f"{lbl}d"

        left_lbl = add_unit(left_lbl)
        right_lbl = add_unit(right_lbl)

        return f"[{left_lbl}, {right_lbl})"

    x_labels = [ _interval_minutes_to_label(iv) for iv in g["delay_bin"] ]

    # ---- Plot ----
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_xlabel(f"{delay_col} (bins in minutes, displayed as days/hours)")

    if title is None:
        title = f"NPS et Volume par {delay_col}"
    ax.set_title(title)

    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return g

```
