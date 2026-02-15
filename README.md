```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_volume_nps_filtered(
    df,
    delay_col,
    score_col,
    delay_breaks,                 # still in MINUTES
    delay_range=None,             # still in MINUTES (min, max)
    promoters_min=9,
    detractors_max=6,
    cat_col=None,
    cat_value=None,
    show_percent=False,
    title=None,
):
    """
    NPS + Volume by delay bins (computed in minutes), with optional categorical filter (cat_col == cat_value).
    X-axis labels are formatted nicely:
      - < 60 min: "15m", "30m"
      - >= 60 min and < 1 day: "2h", "2h 15m"
      - >= 1 day: "3d" or "3d 7h" (no .00)
    """

    # --- keep only needed columns
    cols = [delay_col, score_col]
    if cat_col is not None:
        cols.append(cat_col)
    d = df[cols].copy()

    # --- drop missing delay/score
    d = d.dropna(subset=[delay_col, score_col])

    # --- optional categorical filter
    if cat_col is not None and cat_value is not None:
        d = d[d[cat_col] == cat_value]

    # --- optional delay range filter (minutes)
    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    # --- ensure numeric
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d.dropna(subset=[delay_col, score_col])

    delay_breaks = sorted(delay_breaks)

    # --- binning in minutes
    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    # --- NPS
    def compute_nps(scores: pd.Series):
        if len(scores) == 0:
            return np.nan
        promoters = (scores >= promoters_min).mean() * 100
        detractors = (scores <= detractors_max).mean() * 100
        return promoters - detractors

    # --- aggregate
    g = (
        d.groupby("delay_bin", observed=True)
         .agg(volume=(score_col, "size"), nps=(score_col, compute_nps))
         .reset_index()
    )

    if g.empty:
        print("No data after filters/bins.")
        return g

    # ---------- Pretty label formatting (minutes -> m/h/d) ----------
    def _format_minutes(m):
        """
        m is minutes (float). Output without decimals:
          - 15 -> '15m'
          - 90 -> '1h 30m'
          - 180 -> '3h'
          - 3 days -> '3d'
          - 3.5 days -> '3d 12h'
        """
        total_min = int(round(float(m)))

        # minutes only
        if total_min < 60:
            return f"{total_min}m"

        # compute days/hours/minutes
        day_min = 24 * 60
        days = total_min // day_min
        rem = total_min % day_min
        hours = rem // 60
        mins = rem % 60

        # days
        if days > 0:
            if hours == 0:
                return f"{days}d"
            return f"{days}d {hours}h"  # keep it clean (no minutes with days)

        # hours (no days)
        if mins == 0:
            return f"{hours}h"
        return f"{hours}h {mins}m"

    def _interval_to_label(iv: pd.Interval) -> str:
        return f"[{_format_minutes(iv.left)}, {_format_minutes(iv.right)})"

    x_labels = g["delay_bin"].apply(_interval_to_label).tolist()

    # --- x axis positions
    x = np.arange(len(g))

    # --- bar values (count or percent)
    if show_percent:
        total = g["volume"].sum()
        bar_vals = (g["volume"] / total * 100) if total > 0 else g["volume"] * 0
        bar_ylabel = "Volume (%)"
        bar_fmt = lambda v: f"{v:.1f}%"
    else:
        bar_vals = g["volume"]
        bar_ylabel = "Volume"
        bar_fmt = lambda v: f"{int(v)}"

    # --- title
    if title is None:
        base = f"NPS et Volume par {delay_col}"
        if cat_col is not None and cat_value is not None:
            base += f" | Filtre: {cat_col} = {cat_value}"
        title = base

    # --- plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(x, bar_vals, alpha=0.4)
    ax.set_ylabel(bar_ylabel)

    # bar labels
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            bar_fmt(bar_vals.iloc[i]),
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
    ax.set_xlabel(f"{delay_col} (bins in minutes, affichage m/h/j)")

    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return g

```
