```
import numpy as np
import matplotlib.pyplot as plt

# Use your tables
counts = cozzzunts.copy()
pct = cozzzunts_pct.copy()

# Force correct ranges: KO 0..5, note 0..10
counts = counts.reindex(index=range(0, 6), columns=range(0, 11), fill_value=0)
pct    = pct.reindex(index=range(0, 6), columns=range(0, 11), fill_value=0)

# For plotting: x = notes
notes = list(range(0, 11))
kos   = list(range(0, 6))

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

    # Put text only if segment is big enough (avoids clutter)
    for i, b in enumerate(bars):
        p = vals[i]
        if p >= 6:  # show label only if >= 6%
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



import numpy as np
import matplotlib.pyplot as plt

pct = cozzzunts_pct.copy().reindex(index=range(0, 6), columns=range(0, 11), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))

im = ax.imshow(pct.to_numpy(), aspect="auto")

ax.set_title("Heatmap: % KO par Evaluation Note")
ax.set_xlabel("Evaluation note (0–10)")
ax.set_ylabel("Nombre KO (0–5)")

ax.set_xticks(np.arange(11))
ax.set_xticklabels(range(0, 11))
ax.set_yticks(np.arange(6))
ax.set_yticklabels(range(0, 6))

# Write the % inside each cell (only if > 0.5% to avoid clutter)
for r in range(6):
    for c in range(11):
        v = pct.iloc[r, c]
        if v >= 0.5:
            ax.text(c, r, f"{v:.1f}%", ha="center", va="center", fontsize=8)

plt.colorbar(im, ax=ax, label="Percentage (%)")
plt.tight_layout()
plt.show()
```

    
       