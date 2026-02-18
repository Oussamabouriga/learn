```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_delay_vs_age(
    df: pd.DataFrame,
    delay_col: str,
    age_col: str,
    delay_breaks=None,                 # list of minutes edges (recommended)
    delay_range=None,                  # (min_minutes, max_minutes)
    age_range=None,                    # (min_age, max_age)
    delay_display="auto",              # "auto" | "minutes" | "hours" | "days"
    figsize=(12, 6),
    title=None,
    bar_color="#00177A",               # BLUE3
    alpha=0.9,
    show_age="mean",                   # "mean" | "median"
):
    """
    Bar chart:
      - X: delay bins (minutes -> compact labels)
      - Y: percentage (0..100) of volume in each delay bin
      - Annotate: top of bar = volume %
                  inside bar = age (mean/median)
      - No line plot (removed).
    """

    # ---------- helpers ----------
    def _fmt_minutes(m: float) -> str:
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return ""
        m_int = int(round(float(m)))
        if delay_display == "minutes":
            return f"{m_int}m"
        if delay_display == "hours":
            h = m_int / 60
            return f"{h:.0f}h" if h >= 10 else f"{h:.1f}h"
        if delay_display == "days":
            d = m_int / (24 * 60)
            return f"{d:.0f}d" if d >= 10 else f"{d:.1f}d"

        # auto
        if m_int < 60:
            return f"{m_int}m"
        if m_int < 24 * 60:
            h = m_int / 60
            return f"{h:.0f}h" if h >= 10 else f"{h:.1f}h"
        d = m_int / (24 * 60)
        return f"{d:.0f}d" if d >= 10 else f"{d:.1f}d"

    def _interval_label(iv: pd.Interval) -> str:
        # show like [0m, 1h) / [1d, 3d) etc.
        left = _fmt_minutes(iv.left)
        right = _fmt_minutes(iv.right)
        return f"[{left}, {right})"

    # ---------- validate + clean ----------
    if delay_col not in df.columns:
        raise ValueError(f"Column '{delay_col}' not found in df.")
    if age_col not in df.columns:
        raise ValueError(f"Column '{age_col}' not found in df.")

    d = df[[delay_col, age_col]].copy()
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna(subset=[delay_col, age_col]).copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])].copy()

    if age_range is not None:
        d = d[(d[age_col] >= age_range[0]) & (d[age_col] <= age_range[1])].copy()

    if d.empty:
        raise ValueError("No data left after filtering (check ranges / NaNs).")

    # ---------- bins ----------
    if delay_breaks is None:
        # reasonable default bins in minutes: 0..7d
        max_m = float(d[delay_col].max())
        top = max(1440.0, np.ceil(max_m / 1440.0) * 1440.0)  # round to days
        delay_breaks = [0, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320, 7200, top]

    delay_breaks = sorted(set([float(x) for x in delay_breaks]))
    if len(delay_breaks) < 2:
        raise ValueError("delay_breaks must contain at least 2 edges.")

    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    d = d.dropna(subset=["delay_bin"]).copy()
    if d.empty:
        raise ValueError("All rows fell outside delay_breaks. Expand your delay_breaks/delay_range.")

    # ---------- aggregate ----------
    total = len(d)
    agg = (
        d.groupby("delay_bin", observed=True)
        .agg(
            volume=("delay_bin", "size"),
            age_mean=(age_col, "mean"),
            age_median=(age_col, "median")
        )
        .reset_index()
    )

    agg["pct"] = (agg["volume"] / total) * 100.0
    agg["x_label"] = agg["delay_bin"].apply(_interval_label)

    # keep only non-zero bins (optional)
    agg = agg[agg["volume"] > 0].copy()
    agg = agg.sort_values("delay_bin").reset_index(drop=True)

    # ---------- plot (grid style like your image) ----------
    plt.style.use("ggplot")  # grey background + white grid
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(agg))
    bars = ax.bar(x, agg["pct"].values, color=bar_color, alpha=alpha)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["x_label"].tolist(), rotation=30, ha="right")
    ax.set_xlabel(f"{delay_col} (bins, affichés de façon compacte)")

    if title is None:
        title = f"Répartition du volume par {delay_col} + âge ({show_age})"
    ax.set_title(title)

    # grid exactly on Y like the example
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.15)

    # ---------- annotations ----------
    for i, b in enumerate(bars):
        pct = float(agg["pct"].iloc[i])
        age_val = float(agg["age_mean"].iloc[i] if show_age == "mean" else agg["age_median"].iloc[i])

        # Top label = volume %
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.0,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

        # Inside bar = age (mean/median)
        # put it around mid-height; choose white text when bar is tall
        inside_y = max(1.5, b.get_height() * 0.55)
        txt_color = "white" if b.get_height() >= 18 else "black"
        ax.text(
            b.get_x() + b.get_width() / 2,
            inside_y,
            f"Âge: {age_val:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color=txt_color
        )

    plt.tight_layout()
    plt.show()

    return agg


# Delay must be in minutes in your dataframe
out = plot_delay_vs_age(
    df=df,
    delay_col="delai_Sinistre",
    age_col="Age",
    delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],  # example
    bar_color="#00177A",  # BLUE3
    show_age="mean"
)


```
