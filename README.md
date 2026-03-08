```

# ==============================
# 9) SHAP (FIX robuste CatBoost multi-classes)
# ==============================
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- échantillon pour SHAP
sample_size = min(300, len(X_test_cat_cls))
X_shap = X_test_cat_cls.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_cls_random_best)

# CatBoost multiclass -> shap_values = list[array] (une matrice par classe)
shap_values = explainer.shap_values(X_shap)

# ----- choisir une classe pour le global (ex: 4)
class_for_global = 4 if num_classes > 4 else 0
print("\nSHAP global — classe:", class_for_global, "|", class_names_fr.get(class_for_global, class_for_global))

sv = shap_values[class_for_global]

# sv doit être (n_samples, n_features)
sv = np.asarray(sv)

# FIX 1: parfois une colonne bias en plus -> on enlève la dernière
if sv.shape[1] == X_shap.shape[1] + 1:
    sv = sv[:, :-1]

# FIX 2: sécurité: si encore mismatch, on tronque au min (rare, mais safe)
if sv.shape[1] != X_shap.shape[1]:
    m = min(sv.shape[1], X_shap.shape[1])
    sv = sv[:, :m]
    X_shap_plot = X_shap.iloc[:, :m].copy()
else:
    X_shap_plot = X_shap

# Global beeswarm + bar
shap.summary_plot(sv, X_shap_plot, show=True)
shap.summary_plot(sv, X_shap_plot, plot_type="bar", show=True)


# ==============================
# SHAP local — sur ton exemple
# ==============================
X_one = pd.DataFrame(test_row_cat_cls_no_te).copy()

# align colonnes
for c in X_train_cat_cls.columns:
    if c not in X_one.columns:
        X_one[c] = np.nan
X_one = X_one[X_train_cat_cls.columns].copy()

# même cleanup que dataset (cat -> str + missing label)
X_one = X_one.astype(object).where(pd.notna(X_one), np.nan)

for c in cat_cols_cb:
    if c in X_one.columns:
        X_one[c] = X_one[c].astype(str)
        X_one.loc[X_one[c].isin(["nan", "None", "<NA>"]), c] = "__MISSING__"

for c in num_cols:
    X_one[c] = pd.to_numeric(X_one[c], errors="coerce")

pred_example_class = int(cat_cls_random_best.predict(X_one).reshape(-1)[0])
print("\n=== EXEMPLE ===")
print("Classe prédite:", pred_example_class, "|", class_names_fr.get(pred_example_class, pred_example_class))

# shap values pour l'exemple (list par classe)
shap_one = explainer.shap_values(X_one)
sv_one = np.asarray(shap_one[pred_example_class])

# sv_one doit être (1, n_features)
if sv_one.shape[1] == X_one.shape[1] + 1:
    sv_one = sv_one[:, :-1]

# base value pour la classe prédite
expected = explainer.expected_value
base_val = expected[pred_example_class] if isinstance(expected, (list, np.ndarray)) else expected

# Waterfall clair (top 20)
shap.plots.waterfall(
    shap.Explanation(
        values=sv_one[0],
        base_values=base_val,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()


# ==============================
# 10) Save model (CatBoost)
# ==============================
import os
import json
from datetime import datetime

model_name = "catboost_cls_random_search_no_te_v1"
save_dir = f"models/catboost/classification/{model_name}"
os.makedirs(save_dir, exist_ok=True)

# 1) modèle
model_path = os.path.join(save_dir, "model.cbm")
cat_cls_random_best.save_model(model_path)

# 2) métadonnées (optionnel mais utile)
meta = {
    "model_name": model_name,
    "created_at": datetime.now().isoformat(),
    "num_classes": int(num_classes),
    "class_names_fr": class_names_fr,
    "cat_features": list(cat_cols_cb),
    "best_params": best_params,           # si tu as Random Search
    "best_cv_score_f1_weighted": float(best_cv),
}

with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Model saved to:", model_path)
print("Meta saved to:", os.path.join(save_dir, "meta.json"))
```
