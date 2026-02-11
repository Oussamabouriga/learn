```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_note_distribution_by_delay(
    df: pd.DataFrame,
    delay_col: str,
    note_col: str,                 # evaluate_note
    delay_breaks,                  # custom bins e.g. [0,10,30,60,120,300,600,1500,5000]
    delay_range=None,              # optional (min,max)
    mode="percent",                # "percent" or "count"
    notes_order=None               # optional list like [0,1,...,10]
):
    d = df[[delay_col, note_col]].dropna().copy()

    # optional delay filter
    if delay_range is not None:
        d = d[(d[delay_col] >= delay_range[0]) & (d[delay_col] <= delay_range[1])]

    delay_breaks = sorted(list(delay_breaks))

    # bin delay
    d["delay_bin"] = pd.cut(
        d[delay_col],
        bins=delay_breaks,
        right=False,
        include_lowest=True
    )

    # counts per (delay_bin, note)
    c = (d.groupby(["delay_bin", note_col], observed=True)
           .size()
           .reset_index(name="count"))

    # pivot: rows=delay_bin, cols=note, values=count
    p = (c.pivot_table(index="delay_bin", columns=note_col, values="count", fill_value=0)
           .sort_index())

    # ensure notes order
    if notes_order is None:
        notes_order = sorted(p.columns.tolist())
    else:
        notes_order = [n for n in notes_order if n in p.columns]
    p = p[notes_order]

    # convert to percent distribution within each delay bin
    if mode == "percent":
        denom = p.sum(axis=1).replace(0, np.nan)
        ydata = (p.div(denom, axis=0) * 100)
        ylabel = "Distribution (%)"
    else:
        ydata = p
        ylabel = "Count"

    x = np.arange(len(ydata.index))
    xlabels = [str(iv) for iv in ydata.index]

    # plot: one curve per note
    plt.figure(figsize=(14, 6))
    for note in ydata.columns:
        plt.plot(x, ydata[note].values, marker="o", linewidth=2, label=f"{note_col}={note}")

    plt.xticks(x, xlabels, rotation=30, ha="right")
    plt.xlabel(f"{delay_col} bins")
    plt.ylabel(ylabel)
    plt.title(f"Distribution of {note_col} by {delay_col}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Note", ncol=2)
    plt.tight_layout()
    plt.show()

    return ydata  # table used for plotting


plot_note_distribution_by_delay(
    df,
    delay_col="delai_declaration",
    note_col="evaluate_note",
    delay_breaks=[0, 10, 30, 60, 120, 300, 600, 1500, 5000],
    delay_range=(0, 5000),
    mode="percent",           # or "count"
    notes_order=list(range(0, 11))
)
```
