```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _format_minutes_to_compact(m: float) -> str:
    if pd.isna(m):
        return ""
    m = int(round(float(m)))
    if m < 60:
        return f"{m}m"
    days = m // (24 * 60)
    rem = m % (24 * 60)
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


def _delay_interval_label(left: float, right: float) -> str:
    return f"[{_format_minutes_to_compact(left)}, {_format_minutes_to_compact(right)})"


def _age_interval_label(left: float, right: float) -> str:
    # example: [25,35) -> "25–34"
    l = int(left)
    r = int(right) - 1
    return f"{l}–{r}"


def plot_age_distribution_by_delay_100pct(
    df: pd.DataFrame,
    delay_col: str,
    age_col: str,
    delay_breaks,
    age_breaks,
    title: str = "Répartition des âges par tranches de délai (100%)",
    figsize=(12, 6),
    min_label_pct: float = 6.0,  # don't write text if segment < this %
):
    """
    100% stacked bars:
    - X: delay bins
    - Y: percent 0..100
    - Stack: age bins (dark colors)
    - Text inside: "xx% (n)" for segments big enough
    """

    # --------- clean ----------
    d = df[[delay_col, age_col]].copy()
    d[delay_col] = pd.to_numeric(d[delay_col], errors="coerce")
    d[age_col] = pd.to_numeric(d[age_col], errors="coerce")
    d = d.dropna(subset=[delay_col, age_col])

    delay_breaks = sorted(delay_breaks)
    age_breaks = sorted(age_breaks)

    d["delay_bin"] = pd.cut(d[delay_col], bins=delay_breaks, right=False, include_lowest=True)
    d["age_bin"] = pd.cut(d[age_col], bins=age_breaks, right=False, include_lowest=True)
    d = d.dropna(subset=["delay_bin", "age_bin"])

    # labels (French)
    delay_order = [
        _delay_interval_label(delay_breaks[i], delay_breaks[i + 1])
        for i in range(len(delay_breaks) - 1)
    ]
    age_order = [
        _age_interval_label(age_breaks[i], age_breaks[i + 1])
        for i in range(len(age_breaks) - 1)
    ]

    d["delay_label"] = d["delay_bin"].apply(lambda iv: _delay_interval_label(iv.left, iv.right))
    d["age_label"] = d["age_bin"].apply(lambda iv: _age_interval_label(iv.left, iv.right))

    # --------- counts table ----------
    counts = (
        d.groupby(["delay_label", "age_label"], observed=True)
         .size()
         .unstack(fill_value=0)
         .reindex(index=delay_order, columns=age_order, fill_value=0)
    )

    totals = counts.sum(axis=1).replace(0, np.nan)
    pct = counts.div(totals, axis=0) * 100.0
    pct = pct.fillna(0.0)

    # --------- plot style/colors ----------
    plt.style.use("ggplot")

    # dark palette (blue/purple/gray) – no red/yellow
    dark_palette = [
        "#00177A",  # your BLUE3 (dark blue)
        "#1B2A6B",  # navy-blue
        "#3B3F8C",  # deep indigo
        "#5B4E8C",  # muted purple
        "#4B5563",  # gray
        "#374151",  # darker gray
        "#111827",  # near-black
    ]
    colors = [dark_palette[i % len(dark_palette)] for i in range(len(age_order))]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(delay_order))
    bottom = np.zeros(len(delay_order))

    # --------- stacked bars ----------
    for j, age_lab in enumerate(age_order):
        vals = pct[age_lab].values
        bars = ax.bar(x, vals, bottom=bottom, label=f"Âge {age_lab}", color=colors[j])

        # annotate inside segment
        for i in range(len(delay_order)):
            seg_pct = vals[i]
            seg_n = int(counts.iloc[i, j])
            if seg_n > 0 and seg_pct >= min_label_pct:
                ax.text(
                    x[i],
                    bottom[i] + seg_pct / 2,
                    f"{seg_pct:.0f}%\n({seg_n})",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )

        bottom += vals

    # --------- axis/labels ----------
    ax.set_title(title)
    ax.set_ylabel("Pourcentage (%)")
    ax.set_ylim(0, 100)

    ax.set_xticks(x)
    ax.set_xticklabels(delay_order, rotation=25, ha="right")

    # show total count on top of each bar
    for i, total_n in enumerate(counts.sum(axis=1).fillna(0).astype(int).values):
        ax.text(x[i], 101, f"n={total_n}", ha="center", va="bottom", fontsize=9, color="black")

    ax.legend(title="Tranches d'âge", loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    plt.show()

    # return tables if you want to inspect
    return {"counts": counts, "pct": pct}


```
