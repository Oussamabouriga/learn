
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Column names (edit if needed)
COL_DOSSIER = "dossier_complet"
COL_AI      = "decision_ai"
COL_NOTE    = "evaluate_note"

# ---- 1) Clean needed columns (avoid <NA> issues)
df_plot = df[[COL_DOSSIER, COL_AI, COL_NOTE]].copy()

# Drop missing in these columns
df_plot = df_plot.dropna(subset=[COL_DOSSIER, COL_AI, COL_NOTE])

# Force 0/1 for group cols + int notes
df_plot[COL_DOSSIER] = pd.to_numeric(df_plot[COL_DOSSIER], errors="coerce").astype(int)
df_plot[COL_AI]      = pd.to_numeric(df_plot[COL_AI], errors="coerce").astype(int)
df_plot[COL_NOTE]    = pd.to_numeric(df_plot[COL_NOTE], errors="coerce").astype(int)

# Keep only valid ranges (safety)
df_plot = df_plot[df_plot[COL_DOSSIER].isin([0, 1]) & df_plot[COL_AI].isin([0, 1])]
df_plot = df_plot[df_plot[COL_NOTE].between(0, 10)]

# ---- 2) Build distribution table: rows=(dossier, ai), cols=note(0..10)
counts = (
    df_plot.groupby([COL_DOSSIER, COL_AI])[COL_NOTE]
           .value_counts()
           .unstack(fill_value=0)
)

notes = list(range(0, 11))
counts = counts.reindex(columns=notes, fill_value=0)

# Percentages within each group (row sums to 100)
pct = (counts.div(counts.sum(axis=1), axis=0) * 100).fillna(0)

# ---- 3) Order the 4 groups
groups = [(0,0), (0,1), (1,0), (1,1)]
pct = pct.reindex(groups, fill_value=0)

group_labels = [
    "Dossier non complet - Sans AI",
    "Dossier non complet - Avec AI",
    "Dossier complet - Sans AI",
    "Dossier complet - Avec AI",
]

# ---- 4) Plot (diagramme à barres multiples)
x = np.arange(len(notes))
bar_w = 0.20

plt.figure(figsize=(14, 6))

# nice automatic colors from a colormap
colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))

for i, g in enumerate(groups):
    vals = pct.loc[g].values
    bars = plt.bar(x + (i - 1.5) * bar_w, vals, width=bar_w, label=group_labels[i], color=colors[i])

    # labels on bars (skip tiny)
    for b in bars:
        h = b.get_height()
        if h >= 1.0:
            plt.text(b.get_x() + b.get_width()/2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

plt.xticks(x, notes)
plt.xlabel("Evaluate note (0–10)")
plt.ylabel("Pourcentage (%)")
plt.title("Distribution des notes (0–10) par Dossier complet et Décision AI\n(Diagramme à barres multiples)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


```
