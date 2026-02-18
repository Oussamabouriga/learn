```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_delay_vs_age(
    df,
    age_col,
    delay_col,
    age_breaks=None,
    age_range=(18, 100),

    delay_breaks=None,            # minutes
    delay_range=None,             # (min, max) minutes

    delay_display="auto",         # "auto" | "minutes" | "hours" | "days"
    show_percent=False,
    show_points=True,
    sample_points=5000,

    figsize=(12, 6),
    title=None,

    # ---- style (YOUR COLORS) ----
    blue3="#00177A",              # <-- Blue 3 (requested)
    grid_alpha=0.30,
):
    """
    BLUE3 bars + BLUE3 line, and y-grid like your other plots.
    Delay is stored in minutes, displayed as m/h/d labels.
    """

    # -------- helpers (minutes -> label) --------
    def _format_minutes_auto(m: float) -> str:
        m_int = int(round(float(m)))
        if m_int < 60:
            return f"{m_int}m"
        total = m_int
        days = total // (24 * 60)
        rem = total % (24 * 60)
        hours = rem // 60
        minutes = rem % 60

        if days > 0:
            if hours == 0:
                return f"{days}d"
            return f"{days}d {hours}h"
        if minutes == 0:
            return f"{hours}h"
        return f"{hours}h {minutes}m"

    def _format_minutes(m: float) -> str:
        m = float(m)
        if delay_display == "minutes":
            return f"{int(round(m))}m"
        if delay_display == "hours":
            return f"{m/60:.1f}h"
        if delay_display == "days":
            return f"{m/(60*24):.2f}d"
        return _format_minutes_auto(m)

    # -------- clean data --------
    d = df[[age_col, delay_col]].copy()
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d = d.dropna(subset=[age_col, delay_col])

    if age_range is not None:
        d = d[(d[age_col] >= age_range[0]) & (d[age_col] <= age_range[1])]

    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    if d.empty:
        print("No data after filtering.")
        return None

    # -------- age bins --------
    if age_breaks is None:
        start, end = age_range
        age_breaks = list(range(int(start), int(end) + 1, 5))
        if age_breaks[-1] != int(end):
            age_breaks.append(int(end))

    age_breaks = sorted(age_breaks)
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)

    # -------- aggregate --------
    g = (
        d.groupby("age_bin", observed=True)
         .agg(
             volume=(delay_col, "size"),
             mean_delay=(delay_col, "mean"),
             median_delay=(delay_col, "median"),
         )
         .reset_index()
    )

    x = np.arange(len(g))
    x_labels = g["age_bin"].astype(str).tolist()

    if show_percent:
        total = g["volume"].sum()
        bar_vals = (g["volume"] / total * 100) if total > 0 else g["volume"] * 0
        bar_ylabel = "Volume (%)"
        bar_fmt = lambda v: f"{v:.1f}%"
    else:
        bar_vals = g["volume"]
        bar_ylabel = "Volume"
        bar_fmt = lambda v: f"{int(v)}"

    if title is None:
        title = f"{delay_col} en fonction de {age_col}"

    # -------- plot --------
    fig, ax = plt.subplots(figsize=figsize)

    # Bars (BLUE3)
    bars = ax.bar(
        x,
        bar_vals,
        alpha=0.35,
        color=blue3,
        edgecolor=blue3,
        linewidth=1.0
    )
    ax.set_ylabel(bar_ylabel)

    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            bar_fmt(bar_vals.iloc[i]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    # Line (mean delay) on right axis (BLUE3)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        g["mean_delay"],
        marker="o",
        linewidth=2,
        color=blue3
    )
    ax2.set_ylabel(f"Moyenne de {delay_col}")

    # Optional scatter (BLUE3 transparent)
    if show_points:
        dd = d[[age_col, delay_col]].copy()
        if len(dd) > sample_points:
            dd = dd.sample(sample_points, random_state=42)

        bin_codes = pd.cut(
            dd[age_col],
            bins=age_breaks,
            right=False,
            include_lowest=True
        ).cat.codes.to_numpy()

        ax2.scatter(
            bin_codes,
            dd[delay_col].to_numpy(),
            s=10,
            alpha=0.10,
            color=blue3
        )

    # X axis
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel(age_col)
    ax.set_title(title)

    # Grid style like your plots (y-grid only)
    ax.grid(True, axis="y", alpha=grid_alpha)

    # -------- right axis ticks formatting --------
    if delay_breaks is not None and len(delay_breaks) >= 2:
        delay_breaks = sorted(delay_breaks)
        y_min, y_max = ax2.get_ylim()
        ticks = [t for t in delay_breaks if y_min <= t <= y_max]
        if len(ticks) >= 2:
            ax2.set_yticks(ticks)
            ax2.set_yticklabels([_format_minutes(t) for t in ticks])
    else:
        yt = ax2.get_yticks()
        ax2.set_yticklabels([_format_minutes(t) for t in yt])

    plt.tight_layout()
    plt.show()

    return g



===


	•	Titre de la slide : Distribution des notes de satisfaction et d’évaluation
	•	Titre 1er diagramme : Distribution des notes de satisfaction (0-10)
	•	Titre 2e diagramme : Distribution des notes d’évaluation (IA vs Non-IA)
	•	Conclusion courte : Les notes sont majoritairement hautes (9-10) dans les deux cas. Ce déséquilibre vers le positif peut masquer les raisons de l’insatisfaction. Il faudra se pencher sur les rares notes basses pour mieux les comprendre.

À l’oral, tu peux dire : « On voit que la majorité des notes sont très élevées, ce qui montre un dataset positif, mais on devra analyser les rares notes basses pour comprendre ce qui cause l’insatisfaction. »

Tu n’as qu’à mettre ces titres et cette conclusion courte sur la slide, et le tour est joué !


====


Pour illustrer ta méthodologie sur une slide, tu peux imager un simple schéma linéaire avec des étapes claires. Par exemple :
	1.	Titre de la slide : Méthodologie d’analyse de la satisfaction
	2.	Étapes à visualiser (par exemple, sous forme de flèches) :
	•	Distribution des données : Analyser la répartition des notes et des délais.
	•	NPS par tranches de délai : Étudier la variation du NPS selon les durées de sinistre.
	•	Corrélation & information mutuelle : Identifier les liens entre les variables (ex : délai vs satisfaction).
	•	PCA (future étape) : Réduire la dimension pour prioriser les variables.
	3.	Conclusion visuelle : On suit une approche progressive : on part des données brutes, on mesure l’impact des délais sur le NPS, et on affine la compréhension avec des analyses de relations et de réduction de dimension.

Ainsi, tu donnes une vision claire de ton process et des outils que tu utilises étape par étape.





```
