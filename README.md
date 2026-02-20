```
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- your colors ---
BLACK = "#000000"
BLUE2 = "#728AF4"
BLUE3 = "#00177A"

def _format_minutes_compact(m: float) -> str:
    """Format minutes as compact French label: 45m, 2h, 3j2h, ..."""
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
    if minutes > 0 and days == 0:  # keep short when days exist
        parts.append(f"{minutes}m")
    return "".join(parts) if parts else "0m"

def _interval_to_label(iv: pd.Interval) -> str:
    """Age interval label like [18, 20), [20, 40) ..."""
    if pd.isna(iv):
        return ""
    left = int(iv.left) if float(iv.left).is_integer() else iv.left
    right = int(iv.right) if float(iv.right).is_integer() else iv.right
    return f"[{left}, {right})"

def plot_age_volume_with_delay_stats(
    df: pd.DataFrame,
    age_col: str,
    delay_col: str,                 # delay in minutes
    age_breaks: list,               # e.g. [18, 20, 40, 65, 80, 100]
    delay_range: tuple | None = None,  # e.g. (0, 10080) in minutes
    title: str | None = None,
    figsize=(12, 6),
    bar_color=BLUE3,
    mean_color=BLUE2,
    median_color=BLACK,
):
    """
    Bars = volume (%) by age bins.
    Lines (right axis) = mean + median delay by age bins.
    """
    # --- clean input ---
    use_cols = [age_col, delay_col]
    d = df[use_cols].copy()
    d = d.dropna(subset=[age_col, delay_col])

    # numeric conversion (safe)
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d = d.dropna(subset=[age_col, delay_col])

    if delay_range is not None:
        lo, hi = delay_range
        d = d[(d[delay_col] >= lo) & (d[delay_col] <= hi)]

    # --- binning ---
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

    # --- plot ---
    plt.style.use("ggplot")  # gives the grid style close to your examples
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(g))
    bars = ax.bar(x, g["volume_pct"], color=bar_color, alpha=0.95)

    # Left axis: volume %
    ax.set_ylim(0, 100)
    ax.set_ylabel("Volume (%)")
    ax.set_xlabel("Âge (tranches)")
    ax.set_xticks(x)
    ax.set_xticklabels(g["age_label"], rotation=0)

    # Bar labels: percent + count
    for i, b in enumerate(bars):
        pct = g["volume_pct"].iloc[i]
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

    # Right axis: delay (mean + median)
    ax2 = ax.twinx()
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

    ax2.set_ylabel(f"{delay_col} (m → affichage compact)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: _format_minutes_compact(v)))

    # Grid like your style (clean)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    # Title + spacing
    if title is None:
        title = f"Volume (%) par tranche d'âge + délai (moyen/médian) — {delay_col}"
    ax.set_title(title, pad=18)
    fig.subplots_adjust(top=0.88)

    # Legend top-left (mean+median)
    h1, l1 = ax2.get_legend_handles_labels()
    ax2.legend(h1, l1, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()

    return g


out = plot_age_volume_with_delay_stats(
    df=df,
    age_col="Age",
    delay_col="delai_Sinistre",                 # in minutes
    age_breaks=[18, 20, 40, 65, 80, 100],
    delay_range=(0, 10080),                     # optional: keep 0..7 days
    title="Volume (%) par tranche d’âge + délai moyen/médian"
)

display(out)
```
