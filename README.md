```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _format_minutes_to_compact(m: float) -> str:
    """Convert minutes to compact label: 40m, 2h, 3d4h, etc."""
    if pd.isna(m):
        return ""
    m_int = int(round(float(m)))
    if m_int < 60:
        return f"{m_int}m"
    total = m_int
    days = total // (24 * 60)
    rem = total % (24 * 60)
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


def _interval_minutes_to_label(iv: pd.Interval) -> str:
    left = _format_minutes_to_compact(iv.left)
    right = _format_minutes_to_compact(iv.right)
    return f"[{left}, {right})"


def _interval_age_to_label(iv: pd.Interval) -> str:
    # ex: [25, 35) -> "25–34"
    left = int(iv.left)
    right = int(iv.right) - 1
    return f"{left}–{right}"


def bubble_delay_age_volume(
    df: pd.DataFrame,
    delay_col: str,
    age_col: str,
    delay_breaks,
    age_breaks,
    title: str = "Volume par délai et tranche d'âge",
    figsize=(12, 6),
    alpha=0.45,
    size_scale=35.0,   # bigger => bigger bubbles
):
    """
    Bubble chart:
      - X axis: delay intervals (bins)
      - Y axis: age (midpoint) from 0 to 100
      - Bubble size: volume (count)
      - Bubble label: "count (pct%)"
      - Bubble color: age tranche (legend top-left)
    """

    # ---------- clean data ----------
    d = df[[delay_col, age_col]].dropna().copy()
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna()

    # ---------- bins ----------
    delay_breaks = sorted(delay_breaks)
    age_breaks = sorted(age_breaks)

    d["delay_bin"] = pd.cut(d[delay_col], bins=delay_breaks, right=False, include_lowest=True)
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)

    d = d.dropna(subset=["delay_bin", "age_bin"])

    d["delay_label"] = d["delay_bin"].apply(_interval_minutes_to_label)
    d["age_label"] = d["age_bin"].apply(_interval_age_to_label)

    # Y position = midpoint of age interval
    d["age_mid"] = d["age_bin"].apply(lambda iv: (float(iv.left) + float(iv.right)) / 2.0)

    # ---------- aggregate (IMPORTANT: flat dataframe, no MultiIndex) ----------
    g = (
        d.groupby(["delay_label", "age_label", "age_mid"], observed=True)
         .size()
         .reset_index(name="count")
    )

    # percent within each delay interval (more meaningful than global)
    g["pct"] = g["count"] / g.groupby("delay_label")["count"].transform("sum") * 100.0

    # ---------- order on X ----------
    delay_order = [
        _interval_minutes_to_label(pd.Interval(delay_breaks[i], delay_breaks[i + 1], closed="left"))
        for i in range(len(delay_breaks) - 1)
    ]
    g["delay_label"] = pd.Categorical(g["delay_label"], categories=delay_order, ordered=True)
    g = g.sort_values(["delay_label", "age_mid"])

    x_pos = {lab: i for i, lab in enumerate(delay_order)}
    g["x"] = g["delay_label"].map(x_pos).astype(float)

    # ---------- colors (age categories) ----------
    age_order = sorted(g["age_label"].unique(), key=lambda s: int(str(s).split("–")[0]))
    # use modern colormap access (no deprecation)
    cmap = plt.colormaps.get_cmap("tab10")
    age_to_color = {age_order[i]: cmap(i % 10) for i in range(len(age_order))}

    # ✅ FIX: make sure age_label is scalar strings before mapping
    g["age_label"] = g["age_label"].astype(str)
    g["color"] = g["age_label"].map(age_to_color)

    # ---------- plot ----------
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=figsize)

    sizes = np.maximum(g["count"].values.astype(float), 1.0) * size_scale

    ax.scatter(
        g["x"].values,
        g["age_mid"].values,
        s=sizes,
        c=g["color"].values,
        alpha=alpha,
        edgecolors="none",
    )

    # annotate bubbles: "count (pct%)"
    for _, r in g.iterrows():
        txt = f"{int(r['count'])}\n({r['pct']:.0f}%)"
        ax.text(r["x"], r["age_mid"], txt, ha="center", va="center", fontsize=9, color="black")

    ax.set_title(title)
    ax.set_xlabel("Délai (tranches)")
    ax.set_ylabel("Âge")
    ax.set_ylim(0, 100)

    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels(delay_order, rotation=25, ha="right")

    # legend top-left (age colors)
    handles = [
        Line2D([0], [0], marker="o", color="w", label=f"Âge {lab}", markerfacecolor=age_to_color[lab], markersize=10)
        for lab in age_order
    ]
    ax.legend(handles=handles, title="Tranches d'âge", loc="upper left", frameon=True)

    ax.grid(True, which="major", axis="both", alpha=0.35)
    plt.tight_layout()
    plt.show()

    return g

out = bubble_delay_age_volume(
    df=df,
    delay_col="delai_Sinistre",   # minutes
    age_col="Age",
    delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],
    age_breaks=[0, 25, 35, 45, 60, 120],
    title="Volume par délai et tranche d'âge"
)

```
