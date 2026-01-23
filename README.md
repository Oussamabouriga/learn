```
# --- Distribution: nombre_ko (0..5) by evaluate_note (0..10) ---

import pandas as pd

# 1) Count occurrences for each (evaluate_note, nombre_ko)
dist_ko = (
    df.groupby(["evaluate_note", "nombre_ko"])
      .size()
      .reset_index(name="count")
)

# 2) Percentage within each evaluate_note
dist_ko["percentage"] = (
    dist_ko["count"]
    / dist_ko.groupby("evaluate_note")["count"].transform("sum")
    * 100
).round(2)

# 3) Pivot to a matrix (rows=note, cols=nombre_ko) for plotting
pivot_ko_pct = (
    dist_ko.pivot(index="evaluate_note", columns="nombre_ko", values="percentage")
          .fillna(0)
          .sort_index()
)

# Optional: ensure column order 0..5
pivot_ko_pct = pivot_ko_pct.reindex(columns=range(0, 6), fill_value=0)

pivot_ko_pct

```

```
# --- Better plot (clearer scale + grid + y-axis 0..100 + nicer ticks) ---

import matplotlib.pyplot as plt
import numpy as np

ax = pivot_ko_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 6),
    colormap="Set2",
    width=0.85
)

# Force % scale to be clear
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 10))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)

# Labels / title
ax.set_xlabel("Evaluation note (0–10)")
ax.set_ylabel("Percentage (%)")
ax.set_title("Distribution of Nombre KO by Evaluation Note")

# Make x labels readable
ax.tick_params(axis="x", rotation=0)

# Legend placement
ax.legend(title="Nombre KO", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()


```
