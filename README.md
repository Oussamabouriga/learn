```
import numpy as np
import pandas as pd

counts = cozzzunts.copy()  # rows=KO (0..5), cols=evaluate_note (0..10)

# Column sums (sum of all KO for each evaluate_note)
col_sums = counts.sum(axis=0)

# Percentages by column (each column sums to 100%)
pct_by_note = (counts.div(col_sums, axis=1) * 100).fillna(0)

pct_by_note


import numpy as np
import matplotlib.pyplot as plt

counts = cozzzunts.copy()

# % by column (evaluate_note)
col_sums = counts.sum(axis=0).replace(0, np.nan)
pct_by_note = (counts.div(col_sums, axis=1) * 100).fillna(0)

# For grouped bars: x-axis = notes, bars = KO
counts_T = counts.T          # rows=note, cols=KO
pct_T = pct_by_note.T        # rows=note, cols=KO

counts_arr = counts_T.to_numpy()
pct_arr = pct_T.to_numpy()

notes = counts_T.index.to_list()
kos = counts_T.columns.to_list()

colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

x = np.arange(len(notes))
n_ko = len(kos)
bar_w = 0.12
offsets = (np.arange(n_ko) - (n_ko - 1) / 2) * bar_w

fig, ax = plt.subplots(figsize=(14, 6))

for j, ko in enumerate(kos):
    bars = ax.bar(
        x + offsets[j],
        counts_arr[:, j],
        width=bar_w,
        label=f"KO={ko}",
        color=colors[j % len(colors)],
        edgecolor="black",
        linewidth=0.3
    )

    # write "count + % of that note column"
    for i, b in enumerate(bars):
        c = int(counts_arr[i, j])
        p = float(pct_arr[i, j])
        if c > 0:
            ax.text(
                b.get_x() + b.get_width()/2,
                b.get_height(),
                f"{c}\n({p:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8
            )

ax.set_title("Distribution des KO par note (pourcentage calculé par colonne)")
ax.set_xlabel("evaluate_note")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(notes)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


```

    
       