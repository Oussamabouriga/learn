
```
import numpy as np
import matplotlib.pyplot as plt

# Build counts: rows = (dossier_complet, decision_ai), columns = evaluate_note
counts = (
    df.groupby(["dossier_complet", "decision_ai"])["evaluate_note"]
      .value_counts()
      .unstack(fill_value=0)
)

# Ensure columns 0..10 exist and are ordered
notes = list(range(0, 11))
counts = counts.reindex(columns=notes, fill_value=0)

# Convert to percentages within each group (each row sums to 100)
pct = (counts.div(counts.sum(axis=1), axis=0) * 100)

# Order groups (4 combinations)
groups = [(0,0), (0,1), (1,0), (1,1)]
pct = pct.reindex(groups)

labels = [
    "Dossier non complet - Sans AI",
    "Dossier non complet - Avec AI",
    "Dossier complet - Sans AI",
    "Dossier complet - Avec AI",
]

# ---- Plot grouped bars
x = np.arange(len(notes))        # positions for notes
bar_w = 0.2                      # bar width

plt.figure(figsize=(14, 6))

for i, g in enumerate(groups):
    plt.bar(x + (i - 1.5) * bar_w, pct.loc[g].values, width=bar_w, label=labels[i])

plt.xticks(x, notes)
plt.xlabel("Evaluate note (0–10)")
plt.ylabel("Percentage (%)")
plt.title("Distribution des notes (0–10) par Dossier complet et Décision AI\n(Diagramme à barres multiples)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()


```
