```
import pandas as pd

def dist_binary_table(df, col, labels={0: "0", 1: "1"}):
    s = df[col].dropna().astype(int)
    counts = s.value_counts().reindex([0, 1], fill_value=0)
    pct = (counts / counts.sum() * 100).round(2)

    out = pd.DataFrame({
        col: [labels.get(0, "0"), labels.get(1, "1")],
        "count": counts.values,
        "pct": pct.values
    })
    return out

import matplotlib.pyplot as plt

def dist_binary_plot(df, col, labels={0: "No", 1: "Yes"}, colors=("steelblue", "orange"), figsize=(7,4)):
    t = dist_binary_table(df, col, labels=labels)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(t[col], t["count"], color=list(colors), edgecolor="black", linewidth=0.3)

    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, (c, p) in enumerate(zip(t["count"], t["pct"])):
        ax.text(i, c, f"{c}\n({p}%)", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()
    return t

dist_binary_plot(df, "decision_ai", labels={0:"Sans AI", 1:"Avec AI"})
dist_binary_plot(df, "dossier_complet", labels={0:"Non complet", 1:"Complet"})



```


```
def dist_category_table(df, col, top_n=15):
    s = df[col].fillna("NaN").astype(str).str.strip()
    counts = s.value_counts()

    if top_n is not None and len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    pct = (counts / counts.sum() * 100).round(2)

    out = pd.DataFrame({col: counts.index, "count": counts.values, "pct": pct.values})
    return out


def dist_category_plot(df, col, top_n=15, color="seagreen", figsize=(12,5)):
    t = dist_category_table(df, col, top_n=top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(t[col], t["count"], color=color, edgecolor="black", linewidth=0.3)

    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for i, (c, p) in enumerate(zip(t["count"], t["pct"])):
        ax.text(i, c, f"{c}\n({p}%)", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
    return t


dist_category_plot(df, "PARCOURS_FINAL", top_n=10, color="teal")
dist_category_plot(df, "PARCOURS_INITIAL", top_n=10, color="slateblue")


```


```
import numpy as np

def dist_count_table(df, col, max_bin):
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)

    # cap values > max_bin into one bucket
    capped = s.clip(upper=max_bin)
    counts = capped.value_counts().sort_index()

    # ensure all bins exist 0..max_bin
    counts = counts.reindex(range(0, max_bin+1), fill_value=0)

    # rename last bin as ">=max_bin" (because we clipped)
    idx_labels = [str(i) for i in range(0, max_bin)]
    idx_labels.append(f">={max_bin}")

    pct = (counts / counts.sum() * 100).round(2)

    out = pd.DataFrame({
        col: idx_labels,
        "count": counts.values,
        "pct": pct.values
    })
    return out


def dist_count_plot(df, col, max_bin, color="mediumpurple", figsize=(10,6)):
    t = dist_count_table(df, col, max_bin=max_bin)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(t[col], t["count"], color=color, edgecolor="black", linewidth=0.3)

    ax.set_title(f"Distribution of {col}", fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # labels
    for y, (c, p) in enumerate(zip(t["count"], t["pct"])):
        if c == 0:
            continue
        ax.text(c, y, f" {c} ({p}%)", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()
    return t


dist_count_plot(df, "nombre_prestation_ko", max_bin=5,  color="orange")
dist_count_plot(df, "Nbr_ticket_pieces",    max_bin=14, color="teal")
dist_count_plot(df, "Nbr_ticket_information", max_bin=35, color="slateblue")


```
