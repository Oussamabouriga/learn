```

# ============================================================
# RANDOM FOREST CLASSIFICATION — SHAP FIX (GLOBAL + EXEMPLE) + SAVE
# ============================================================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os, json
from datetime import datetime
import joblib

# -----------------------------
# 0) Choisis ton modèle RF + tes données numériques
# -----------------------------
rf_cls_model = rf_cls_model

# IMPORTANT: X_test_rf doit être NUMÉRIQUE (one-hot / freq déjà fait)
X_test_rf = X_test_rf.copy().astype(float)   # adapte le nom: X_test_xgb_cls_base etc.
X_train_rf = X_train_rf.copy().astype(float)

# Exemple déjà encodé (doit matcher les colonnes RF)
X_one_rf = X_one_rf.copy().astype(float)     # adapte le nom si besoin

# -----------------------------
# 1) SHAP global
# -----------------------------
sample_size = min(300, len(X_test_rf))
X_shap = X_test_rf.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(rf_cls_model)
shap_values = explainer.shap_values(X_shap)

n_classes = len(rf_cls_model.classes_)
class_for_global = 4 if n_classes > 4 else 0

if isinstance(shap_values, list):
    sv = np.asarray(shap_values[class_for_global])
else:
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        sv = sv[:, :, class_for_global]

print("RF X_shap:", X_shap.shape, "| RF sv:", sv.shape)

# FIX bias column if exists
if sv.shape[1] == X_shap.shape[1] + 1:
    sv = sv[:, :-1]

if sv.shape[1] != X_shap.shape[1]:
    m = min(sv.shape[1], X_shap.shape[1])
    sv = sv[:, :m]
    X_shap_plot = X_shap.iloc[:, :m].copy()
else:
    X_shap_plot = X_shap

shap.summary_plot(sv, X_shap_plot, show=True)
shap.summary_plot(sv, X_shap_plot, plot_type="bar", show=True)

# -----------------------------
# 2) SHAP local (exemple)
# -----------------------------
pred_class = int(rf_cls_model.predict(X_one_rf)[0])
print("Classe prédite (exemple RF):", pred_class)

shap_one = explainer.shap_values(X_one_rf)

if isinstance(shap_one, list):
    sv_one = np.asarray(shap_one[pred_class])
else:
    sv_one = np.asarray(shap_one)
    if sv_one.ndim == 3:
        sv_one = sv_one[:, :, pred_class]

if sv_one.shape[1] == X_one_rf.shape[1] + 1:
    sv_one = sv_one[:, :-1]

expected = explainer.expected_value
base_val = expected[pred_class] if isinstance(expected, (list, np.ndarray)) else expected

shap.plots.waterfall(
    shap.Explanation(
        values=sv_one[0],
        base_values=base_val,
        data=X_one_rf.iloc[0],
        feature_names=X_one_rf.columns
    ),
    max_display=20
)
plt.show()

# -----------------------------
# 3) SAVE RF model
# -----------------------------
model_name = "random_forest_cls_v1"
save_dir = f"models/random_forest/classification/{model_name}"
os.makedirs(save_dir, exist_ok=True)

path_model = os.path.join(save_dir, "model.joblib")
joblib.dump(rf_cls_model, path_model)

meta = {
    "model_name": model_name,
    "created_at": datetime.now().isoformat(),
    "classes": [int(x) for x in rf_cls_model.classes_],
    "columns": list(X_train_rf.columns),
}
with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Saved:", path_model)
print("Saved:", os.path.join(save_dir, "meta.json"))
```
