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
import matplotlib.pyplot as plt

colors = ["green", "lightgreen", "yellow", "orange", "red", "purple"]  # for KO 0..5

ax = pivot_ko_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
    color=colors
)

plt.xlabel("Evaluation note (0–10)")
plt.ylabel("Percentage (%)")
plt.title("Distribution of Nombre KO by Evaluation Note")
plt.legend(title="Nombre KO", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()



import matplotlib.colors as mcolors
list(mcolors.CSS4_COLORS.keys())[:30]


import pandas as pd

# Count occurrences for each (evaluate_note, nombre_ko)
dist_ko_count = (
    df.groupby(["evaluate_note", "nombre_ko"])
      .size()
      .reset_index(name="count")
)

# Pivot to matrix: rows = evaluate_note, columns = nombre_ko
pivot_ko_count = (
    dist_ko_count.pivot(
        index="evaluate_note",
        columns="nombre_ko",
        values="count"
    )
    .fillna(0)
    .astype(int)
    .sort_index()
)

# Ensure columns 0..5 exist and ordered
pivot_ko_count = pivot_ko_count.reindex(columns=range(0, 6), fill_value=0)

pivot_ko_count



from tabulate import tabulate

# Move index to a column for stable display
pivot_ko_table = pivot_ko_count.reset_index()
pivot_ko_table.rename(columns={"evaluate_note": "Evaluation note"}, inplace=True)

print(
    tabulate(
        pivot_ko_table,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )
)



```
