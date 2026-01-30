```

import matplotlib.pyplot as plt
import numpy as np

# transpose -> rows=evaluate_note, cols=KO (easier for grouped bars)
counts_T = counts.T
pct_T = pct.T

# choose your colors by name (KO 0..5)
colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

x = np.arange(len(counts_T.index))          # evaluate_note positions
n_ko = len(counts_T.columns)                # number of KO categories
bar_w = 0.12                                # bar width
offsets = (np.arange(n_ko) - (n_ko - 1) / 2) * bar_w

fig, ax = plt.subplots(figsize=(14, 6))

for j, ko in enumerate(counts_T.columns):
    bars = ax.bar(
        x + offsets[j],
        counts_T[ko].values,
        width=bar_w,
        label=f"KO={ko}",
        color=colors[j % len(colors)],
        edgecolor="black",
        linewidth=0.3
    )

    # labels: count + % (skip tiny bars to avoid clutter)
    for i, b in enumerate(bars):
        c = int(counts_T.iloc[i, j])
        p = float(pct_T.iloc[i, j])
        if c > 0:  # show only non-zero
            ax.text(
                b.get_x() + b.get_width()/2,
                b.get_height(),
                f"{c}\n({p:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )

ax.set_title("Nombre KO par Evaluation Note (barres multiples)")
ax.set_xlabel("Evaluation note")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(counts_T.index)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```

    
       