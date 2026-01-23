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
# --- Plot: stacked percentage bars (best to understand distribution) ---

import matplotlib.pyplot as plt

ax = pivot_ko_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    colormap="Set2"
)

plt.xlabel("Evaluation note (0–10)")
plt.ylabel("Percentage (%)")
plt.title("Distribution of Nombre KO by Evaluation Note")
plt.legend(title="Nombre KO", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


```
