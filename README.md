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
    X-axis labels are shown in MINUTES / HOURS / DAYS depending on size,
    but binning stays in minutes.
    """

    d = df[[delay_col, score_col]].dropna().copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    delay_breaks = sorted(delay_breaks)

    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    def compute_nps(scores: pd.Series) -> float:
        if len(scores) == 0:
            return np.nan
        promoters = (scores >= promoters_min).mean() * 100
        detractors = (scores <= detractors_max).mean() * 100
        return promoters - detractors

    g = (
        d.groupby("delay_bin", observed=True)
         .agg(volume=(delay_col, "size"),
              nps=(score_col, compute_nps))
         .reset_index()
    )

    # ---------- Formatting helpers (display only) ----------
    def _format_minutes_to_compact(m: float) -> str:
        """
        Rules:
        - < 60 min: show "15m", "30m" (NO decimals)
        - 60..(24h-1): show hours like "3h" or "3h 15m" (no seconds)
        - >= 24h: show "3d" or "3d 2h" or "3d 2h 15m"
        """
        m_int = int(round(m))
        if m_int < 60:
            return f"{m_int}m"

        total_minutes = m_int
        days = total_minutes // (24 * 60)
        rem = total_minutes % (24 * 60)
        hours = rem // 60
        minutes = rem % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 and days == 0:  # show minutes with hours, but keep days labels clean
            parts.append(f"{minutes}m")
        elif minutes > 0 and days > 0:
            # if you also want minutes even with days, uncomment next line:
            # parts.append(f"{minutes}m")
            pass

        return " ".join(parts) if parts else "0m"

    def _interval_minutes_to_label(iv: pd.Interval) -> str:
        left_lbl = _format_minutes_to_compact(iv.left)
        right_lbl = _format_minutes_to_compact(iv.right)
        return f"[{left_lbl}, {right_lbl})"

    x_labels = [_interval_minutes_to_label(iv) for iv in g["delay_bin"]]

    # ---------- Plot ----------
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(x, g["volume"], alpha=0.4)
    ax.set_ylabel("Volume")

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(g['volume'].iloc[i])}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax2 = ax.twinx()
    ax2.plot(x, g["nps"], marker="o", linewidth=2)
    ax2.set_ylabel("NPS")
    ax2.set_ylim(-100, 100)

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
    ax.set_xlabel(f"{delay_col} (bins in minutes, displayed compactly)")

    if title is None:
        title = f"NPS et Volume par {delay_col}"
    ax.set_title(title)

    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return g
```
