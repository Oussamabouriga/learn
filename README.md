```

# ==============================
# Class names (index -> label)
# ==============================
class_names = [
    "Extrêmement mauvais (0–2)",
    "Mauvais (3–6)",
    "Neutre (7–8)",
    "Bien (9)",
    "Très bien (10)",
]

num_classes = len(class_names)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

for c in range(num_classes):
    y_true_bin = (y_test_xgb_cls_base.values == c).astype(int)
    fpr, tpr, _ = roc_curve(y_true_bin, proba_test[:, c])
    plt.plot(fpr, tpr, label=class_names[c])

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Courbes ROC (One-vs-Rest) — jeu de test")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




from sklearn.metrics import confusion_matrix
import pandas as pd

cm = confusion_matrix(y_test_xgb_cls_base, pred_test_cls, labels=list(range(num_classes)))

cm_df = pd.DataFrame(
    cm,
    index=[f"Réel — {name}" for name in class_names],
    columns=[f"Prédit — {name}" for name in class_names],
)

print("\nMatrice de confusion (test)")
display(cm_df)
```
