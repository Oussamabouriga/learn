```

import numpy as np
import matplotlib.pyplot as plt

# Use your tables
counts = cozzzunts.copy()
pct = cozzzunts_pct.copy()

# Force correct ranges: KO 0..5, note 0..10
counts = counts.reindex(index=range(0, 6), columns=range(0, 11), fill_value=0)
pct    = pct.reindex(index=range(0, 6), columns=range(0, 11), fill_value=0)

# notes and kos (no list() to avoid the error)
notes = np.arange(0, 11)
kos   = np.arange(0, 6)

# transpose so rows=notes, cols=KO
pct_T = pct.T
cnt_T = counts.T

fig, ax = plt.subplots(figsize=(14, 6))

bottom = np.zeros(len(notes))
colors = ["green", "lightgreen", "gold", "orange", "red", "black"]

for j, ko in enumerate(kos):
    vals = pct_T[ko].to_numpy()

    bars = ax.bar(
        notes, vals,
        bottom=bottom,
        label=f"KO={ko}",
        color=colors[j],
        edgecolor="white",
        linewidth=0.5
    )

    # labels: show only if segment >= 6% to stay clean
    for i, b in enumerate(bars):
        p = vals[i]
        if p >= 6:
            c = int(cnt_T.iloc[i, j])
            ax.text(
                b.get_x() + b.get_width()/2,
                bottom[i] + p/2,
                f"{p:.1f}%\n({c})",
                ha="center", va="center",
                fontsize=8, color="black"
            )

    bottom += vals

ax.set_title("Distribution de KO par Evaluation Note (100% stacked) — % + (count)")
ax.set_xlabel("Evaluation note (0–10)")
ax.set_ylabel("Percentage (%)")
ax.set_xticks(notes)
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
```

    
       