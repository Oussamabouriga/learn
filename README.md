```
import numpy as np
import matplotlib.pyplot as plt

# --- INPUTS (already in your notebook) ---
# cozzzunts      -> counts (rows=KO 0..5, cols=evaluate_note 0..10)
# cozzzunts_pct  -> percentages (same shape)

counts = cozzzunts.copy()
pct = cozzzunts_pct.copy()

# Ensure same order
counts = counts.sort_index().sort_index(axis=1)
pct = pct.reindex(index=counts.index, columns=counts.columns)

# For grouped bars: x-axis = evaluate_note, bars = KO
counts_T = counts.T   # rows=note, cols=KO
pct_T = pct.T         # rows=note, cols=KO

notes = counts_T.index.to_list()
kos = counts_T.columns.to_list()

counts_arr = counts_T.to_numpy()
pct_arr = pct_T.to_numpy()

# Colors (change if you want)
colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

x = np.arange(len(notes))
n_ko = len(kos)
bar_w = 0.12
offsets = (np.arange(n_ko) - (n_ko - 1) / 2) * bar_w

# Adaptive size (depends on number of notes)
fig_w = max(12, len(notes) * 1.2)
fig, ax = plt.subplots(figsize=(fig_w, 6))

for j, ko in enumerate(kos):
    bars = ax.bar(
        x + offsets[j],
        counts_arr[:, j],                 # bar HEIGHT = COUNT
        width=bar_w,
        label=f"KO={ko}",
        color=colors[j % len(colors)],
        edgecolor="black",
        linewidth=0.3
    )

    # Labels: "count (xx.x%)"
    for i, b in enumerate(bars):
        c = int(counts_arr[i, j])
        p = float(pct_arr[i, j])
        if c > 0:
            ax.text(
                b.get_x() + b.get_width()/2,
                b.get_height(),
                f"{c}\n({p:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=8
            )

ax.set_title("Nombre KO par Evaluation Note (grouped bars) — counts + %")
ax.set_xlabel("Evaluation note")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(notes)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

    
       