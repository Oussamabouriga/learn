```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_volume_nps(
    df,
    delay_col,
    score_col,
    delay_breaks,
    delay_range=None,
    promoters_min=9,
    detractors_max=6,
    figsize=(12, 6),
    title=None,
):
    # --- data prep ---
    d = df[[delay_col, score_col]].dropna().copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])].copy()

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
         .agg(
             volume=(delay_col, "size"),
             nps=(score_col, compute_nps),
         )
         .reset_index()
    )

    # --- pretty x labels ---
    def _format_minutes_to_compact(m: float) -> str:
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
        if minutes > 0 and days == 0:
            parts.append(f"{minutes}m")
        return "".join(parts) if parts else "0m"

    def _interval_minutes_to_label(iv: pd.Interval) -> str:
        left_lbl = _format_minutes_to_compact(iv.left)
        right_lbl = _format_minutes_to_compact(iv.right)
        return f"[{left_lbl}, {right_lbl})"

    x_labels = [_interval_minutes_to_label(iv) for iv in g["delay_bin"]]

    # --- plot (ggplot background + white grid like your image) ---
    with plt.style.context("ggplot"):
        x = np.arange(len(g))

        fig, ax = plt.subplots(figsize=figsize)

        # Force BLUE (ggplot would otherwise start red/orange)
        bars = ax.bar(x, g["volume"], alpha=0.4, color="tab:blue")
        ax.set_ylabel("Volume")

        # bar labels
        for i, b in enumerate(bars):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(g['volume'].iloc[i])}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        # NPS line (BLUE)
        ax2 = ax.twinx()
        ax2.plot(x, g["nps"], marker="o", linewidth=2, color="tab:blue")
        ax2.set_ylabel("NPS")
        ax2.set_ylim(-100, 100)

        # ✅ Keep same “size/density” as when it was 25 by 25 (labels every 25)
        ax2.yaxis.set_major_locator(MultipleLocator(25))

        # ✅ But add “10 by 10” resolution (minor ticks/grid every 10)
        ax2.yaxis.set_minor_locator(MultipleLocator(10))

        # Grid styling (like your screenshot: clear horizontal grid)
        ax.grid(True, axis="y", which="major", alpha=0.35)
        ax2.grid(True, axis="y", which="minor", alpha=0.18)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_xlabel(f"{delay_col} (bins in minutes, displayed compactly)")

        if title is None:
            title = f"NPS et Volume par {delay_col}"
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    return g

```
