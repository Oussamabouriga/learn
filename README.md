```

                      import pandas as pd

# Count matrix: rows=evaluate_note (0..10), cols=nombre_ko (0..5)
pivot_ko_count = (
    df.groupby(["evaluate_note", "nombre_ko"])
      .size()
      .unstack(fill_value=0)
      .reindex(index=range(0, 11), fill_value=0)   # ensure notes 0..10
      .reindex(columns=range(0, 6), fill_value=0)  # ensure ko 0..5
)

# Percentages per evaluate_note row (safe division)
pivot_ko_pct = (pivot_ko_count.div(pivot_ko_count.sum(axis=1), axis=0) * 100).fillna(0)

pivot_ko_count, pivot_ko_pct



import matplotlib.pyplot as plt

# choose your colors by name (one per KO level 0..5)
colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

ax = pivot_ko_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 7),
    color=colors
)

plt.title("Distribution de Nombre KO (0..5) pour chaque Evaluation Note (0..10)")
plt.xlabel("Evaluation note")
plt.ylabel("Pourcentage (%)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.legend(title="Nombre KO", bbox_to_anchor=(1.02, 1), loc="upper left")

# add labels: count + %
for i, note in enumerate(pivot_ko_pct.index):
    cumulative = 0
    total = pivot_ko_count.loc[note].sum()
    for ko in pivot_ko_pct.columns:
        pct = pivot_ko_pct.loc[note, ko]
        cnt = pivot_ko_count.loc[note, ko]
        if pct > 3:  # avoid clutter for tiny segments
            y = cumulative + pct / 2
            ax.text(i, y, f"{cnt}\n({pct:.1f}%)", ha="center", va="center", fontsize=8)
        cumulative += pct

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import numpy as np

colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

notes = pivot_ko_count.index.to_list()
kos = pivot_ko_count.columns.to_list()

x = np.arange(len(notes))
width = 0.13  # 6 bars per group

fig, ax = plt.subplots(figsize=(16, 6))

for j, ko in enumerate(kos):
    heights = pivot_ko_count[ko].values
    bars = ax.bar(x + (j - (len(kos)-1)/2)*width, heights, width, label=f"KO={ko}", color=colors[j])

    # labels like "count (pct%)"
    for b, note in zip(bars, notes):
        cnt = pivot_ko_count.loc[note, ko]
        pct = pivot_ko_pct.loc[note, ko]
        if cnt > 0:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{cnt}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=7)

ax.set_title("Nombre KO par Evaluation Note (grouped bars)")
ax.set_xlabel("Evaluation note")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(notes)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
    """

    
       