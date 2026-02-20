```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Optional, Tuple, List

BLACK = "#000000"
BLUE2 = "#728AF4"
BLUE3 = "#00177A"

def _format_minutes_compact(m: float) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return ""
    m = float(m)
    if m < 60:
        return f"{int(round(m))}m"

    total_minutes = int(round(m))
    days = total_minutes // (24 * 60)
    rem = total_minutes % (24 * 60)
    hours = rem // 60
    minutes = rem % 60

    parts = []
    if days > 0:
        parts.append(f"{days}j")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 and days == 0:
        parts.append(f"{minutes}m")
    return "".join(parts) if parts else "0m"

def _interval_to_label(iv: Optional[pd.Interval]) -> str:
    if iv is None or pd.isna(iv):
        return ""
    left = int(iv.left) if float(iv.left).is_integer() else iv.left
    right = int(iv.right) if float(iv.right).is_integer() else iv.right
    return f"[{left}, {right})"

def plot_age_volume_with_delay_stats(
    df: pd.DataFrame,
    age_col: str,
    delay_col: str,
    age_breaks: List[float],
    delay_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    bar_color: str = BLUE3,
    mean_color: str = BLUE2,
    median_color: str = BLACK,
):
    use_cols = [age_col, delay_col]
    d = df[use_cols].copy()

    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d = d.dropna(subset=[age_col, delay_col])

    if delay_range is not None:
        lo, hi = delay_range
        d = d[(d[delay_col] >= lo) & (d[delay_col] <= hi)]

    age_breaks = sorted(age_breaks)
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)

    g = (
        d.groupby("age_bin", observed=True)
        .agg(
            volume=("age_bin", "size"),
            delay_mean=(delay_col, "mean"),
            delay_median=(delay_col, "median"),
        )
        .reset_index()
    )

    total = int(g["volume"].sum())
    if total == 0:
        raise ValueError("No data after filtering/bins. Check age_breaks and delay_range.")

    g["volume_pct"] = g["volume"] / total * 100.0
    g["age_label"] = g["age_bin"].apply(_interval_to_label)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(g))
    bars = ax.bar(x, g["volume_pct"], color=bar_color, alpha=0.95)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Volume (%)")
    ax.set_xlabel("Âge (tranches)")
    ax.set_xticks(x)
    ax.set_xticklabels(g["age_label"], rotation=0)

    for i, b in enumerate(bars):
        pct = float(g["volume_pct"].iloc[i])
        n = int(g["volume"].iloc[i])
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.5,
            f"{pct:.0f}%\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BLACK,
        )

    ax2 = ax.twinx()

    # --- lines ---
    ax2.plot(
        x, g["delay_mean"],
        color=mean_color, marker="o", linewidth=2,
        label="Délai moyen"
    )
    ax2.plot(
        x, g["delay_median"],
        color=median_color, marker="o", linewidth=2,
        label="Délai médian"
    )

    # ✅ ADD TEXT LABELS on each point (same color as each line)
    for i, val in enumerate(g["delay_mean"].values):
        if pd.notna(val):
            ax2.annotate(
                _format_minutes_compact(val),
                (x[i], val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color=mean_color,
            )

    for i, val in enumerate(g["delay_median"].values):
        if pd.notna(val):
            ax2.annotate(
                _format_minutes_compact(val),
                (x[i], val),
                textcoords="offset points",
                xytext=(0, -14),   # slightly lower so it doesn't overlap the mean label
                ha="center",
                fontsize=9,
                color=median_color,
            )

    ax2.set_ylabel(f"{delay_col} (affichage compact)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: _format_minutes_compact(v)))

    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    if title is None:
        title = f"Volume (%) par tranche d'âge + délai (moyen/médian) — {delay_col}"
    ax.set_title(title, pad=22)
    fig.subplots_adjust(top=0.86)

    h1, l1 = ax2.get_legend_handles_labels()
    ax2.legend(h1, l1, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()

    return g


```
