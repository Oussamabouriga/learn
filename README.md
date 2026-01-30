```


import pandas as pd
import matplotlib.pyplot as plt

# 1) Table de % par evaluate_note (chaque ligne = 100%)
pct = (
    pd.crosstab(df["evaluate_note"], df["nombre_prestation_ko"], normalize="index")
      .mul(100)
      .round(2)
)

# (optionnel) trier les colonnes KO dans l'ordre
pct = pct.reindex(sorted(pct.columns), axis=1)

# 2) Plot 100% stacked
ax = pct.plot(kind="bar", stacked=True, figsize=(10, 5))

ax.set_title("Distribution (%) de nombre_prestation_ko par evaluate_note")
ax.set_xlabel("evaluate_note")
ax.set_ylabel("Pourcentage (%)")
ax.set_ylim(0, 100)
ax.legend(title="nombre_prestation_ko", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()
```

    
       