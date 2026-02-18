```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def bubble_delay_age_volume(
    df: pd.DataFrame,
    delay_col: str,
    age_col: str,
    delay_breaks,                 # ex: [0, 60, 120, 240, 480, 1440, ...] (minutes)
    age_breaks=None,              # ex: [0, 25, 35, 45, 60, 120] or None -> auto (quantiles)
    delay_range=None,             # (min, max) optional
    age_range=None,               # (min, max) optional
    title="Volume par délai et tranche d'âge",
    figsize=(13, 6),
    max_bubble_area=4500,         # controls bubble size (avoid huge figures)
    min_bubble_area=200,
    alpha=0.55,
    edgecolor="white",
    linewidth=1.2,
    show_grid=True,
    fmt_delay="auto",             # "auto" | "minutes" | "hours" | "days"
):
    """
    Bubble chart (like your example):
      - X = intervalle de délai (bins)
      - Y = tranche d'âge (bins)
      - Taille de bulle = volume (count)
      - Texte dans la bulle = "count\n(p%)"
      - Couleur = tranche d'âge (légende en haut à gauche)

    Returns:
      pivot table with count/pct per (delay_bin, age_bin)
    """

    # ---------------- helpers ----------------
    def _fmt_minutes(m: float) -> str:
        m = float(m)
        if fmt_delay == "minutes":
            return f"{int(round(m))} min"
        if fmt_delay == "hours":
            h = m / 60
            return f"{h:.0f} h" if h >= 10 else f"{h:.1f} h"
        if fmt_delay == "days":
            d = m / (24 * 60)
            return f"{d:.0f} j" if d >= 10 else f"{d:.1f} j"

        # auto
        if m < 60:
            return f"{int(round(m))} min"
        if m < 24 * 60:
            h = m / 60
            return f"{h:.0f} h" if h >= 10 else f"{h:.1f} h"
        d = m / (24 * 60)
        return f"{d:.0f} j" if d >= 10 else f"{d:.1f} j"

    def _interval_label(iv: pd.Interval) -> str:
        return f"[{_fmt_minutes(iv.left)} ; {_fmt_minutes(iv.right)})"

    # ---------------- validate + clean ----------------
    if delay_col not in df.columns:
        raise ValueError(f"Colonne '{delay_col}' introuvable.")
    if age_col not in df.columns:
        raise ValueError(f"Colonne '{age_col}' introuvable.")
    if delay_breaks is None or len(delay_breaks) < 2:
        raise ValueError("delay_breaks doit contenir au moins 2 bornes.")

    d = df[[delay_col, age_col]].copy()
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna(subset=[delay_col, age_col]).copy()

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])].copy()

    if age_range is not None:
        d = d[(d[age_col] >= age_range[0]) & (d[age_col] <= age_range[1])].copy()

    if d.empty:
        raise ValueError("Aucune donnée après filtrage (delay_range/age_range/NaN).")

    delay_breaks = sorted(set([float(x) for x in delay_breaks]))

    # auto age bins (quantiles) if not provided
    if age_breaks is None:
        qs = d[age_col].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        age_breaks = np.unique(np.round(qs, 0)).astype(float).tolist()
        if len(age_breaks) < 3:
            mn, mx = float(d[age_col].min()), float(d[age_col].max())
            age_breaks = [mn, (mn + mx) / 2, mx]

    age_breaks = sorted(set([float(x) for x in age_breaks]))
    if len(age_breaks) < 2:
        raise ValueError("age_breaks doit contenir au moins 2 bornes.")

    # ---------------- binning ----------------
    d["delay_bin"] = pd.cut(d[delay_col], bins=delay_breaks, right=False, include_lowest=True)
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)

    d = d.dropna(subset=["delay_bin", "age_bin"]).copy()
    if d.empty:
        raise ValueError("Toutes les lignes sont hors des bornes (delay_breaks/age_breaks).")

    # ---------------- aggregate ----------------
    total = len(d)
    g = (
        d.groupby(["delay_bin", "age_bin"], observed=True)
        .size()
        .reset_index(name="volume")
    )
    g["pct"] = (g["volume"] / total) * 100.0

    # labels
    delay_labels = {iv: _interval_label(iv) for iv in sorted(g["delay_bin"].unique())}
    age_labels = {iv: f"[{int(iv.left)} ; {int(iv.right)}) ans" for iv in sorted(g["age_bin"].unique())}
    g["delay_label"] = g["delay_bin"].map(delay_labels)
    g["age_label"] = g["age_bin"].map(age_labels)

    # numeric positions for plotting
    delay_order = list(delay_labels.values())
    age_order = list(age_labels.values())

    g["x"] = g["delay_label"].apply(lambda s: delay_order.index(s))
    g["y"] = g["age_label"].apply(lambda s: age_order.index(s))

    # ---------------- bubble size scaling ----------------
    vmin, vmax = g["volume"].min(), g["volume"].max()
    if vmax == vmin:
        g["area"] = (min_bubble_area + max_bubble_area) / 2
    else:
        g["area"] = min_bubble_area + (g["volume"] - vmin) * (max_bubble_area - min_bubble_area) / (vmax - vmin)

    # ---------------- color per age group ----------------
    # use a discrete colormap, one color per age bin
    cmap = plt.cm.get_cmap("tab10", len(age_order))
    age_to_color = {age_order[i]: cmap(i) for i in range(len(age_order))}
    g["color"] = g["age_label"].map(age_to_color)

    # ---------------- plot ----------------
    plt.style.use("ggplot")  # gives a grid vibe similar to your slides; remove if you want pure white
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        g["x"],
        g["y"],
        s=g["area"],
        c=g["color"].tolist(),
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=linewidth,
    )

    # text inside bubbles: "count\n(pct%)"
    for _, row in g.iterrows():
        ax.text(
            row["x"],
            row["y"],
            f"{int(row['volume'])}\n({row['pct']:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
        )

    # axes
    ax.set_xticks(range(len(delay_order)))
    ax.set_xticklabels(delay_order, rotation=20, ha="right")
    ax.set_yticks(range(len(age_order)))
    ax.set_yticklabels(age_order)

    ax.set_xlabel("Délai (intervalles)")
    ax.set_ylabel("Tranches d'âge")
    ax.set_title(title)

    if show_grid:
        ax.grid(True, alpha=0.25)
    else:
        ax.grid(False)

    # legend top-left (colors = age groups)
    handles = [Patch(facecolor=age_to_color[a], edgecolor="none", label=a) for a in age_order]
    ax.legend(handles=handles, title="Âge (code couleur)", loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()

    # return a pivot-style table for your report if needed
    out = g[["delay_label", "age_label", "volume", "pct"]].copy()
    return out


# ---------------- Example call ----------------
# out = bubble_delay_age_volume(
#     df=df,
#     delay_col="delai_Sinistre",      # minutes
#     age_col="Age",
#     delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],  # example
#     age_breaks=[0, 25, 35, 45, 60, 120],                                  # example
#     title="Volume par délai et âge (comptes et pourcentages)"
# )
# display(out.head())


out = bubble_delay_age_volume(
    df=df,
    delay_col="delai_Sinistre",   # in minutes
    age_col="Age",
    delay_breaks=[0, 60, 120, 240, 480, 1440, 2880, 4320, 7200, 10080],
    age_breaks=[0, 25, 35, 45, 60, 120],
    title="Volume par délai et tranche d'âge"
)

```
