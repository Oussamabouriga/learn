```
# ============================================================
# XGBoost Regressor (NO Target Encoding, NO optimization)
# + SAVE the model to: models/xgboost/regression/<unique_model_name>/
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

# -----------------------------
# 0) Data (no_te)
# -----------------------------
X_train_xgb_reg_no_te = X_train_encoded_no_te.copy().astype(float)
X_test_xgb_reg_no_te  = X_test_encoded_no_te.copy().astype(float)

y_train_xgb_reg_no_te = pd.to_numeric(y_train_no_te, errors="coerce").astype(float)
y_test_xgb_reg_no_te  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(float)

print("Train:", X_train_xgb_reg_no_te.shape, y_train_xgb_reg_no_te.shape)
print("Test :", X_test_xgb_reg_no_te.shape,  y_test_xgb_reg_no_te.shape)

# -----------------------------
# 1) Hyperparameters (dict)
# -----------------------------
xgb_reg_params_no_te = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",

    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "gamma": 0.0,

    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
    "missing": np.nan,
}

# -----------------------------
# 2) Train
# -----------------------------
xgboost_reg_no_te = XGBRegressor(**xgb_reg_params_no_te)
xgboost_reg_no_te.fit(X_train_xgb_reg_no_te, y_train_xgb_reg_no_te)
print("✅ Modèle XGBoost Regressor (no_te) entraîné")

# -----------------------------
# 3) SAVE model (unique name) + metadata
# -----------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
unique_model_name = f"xgb_reg_no_te_{timestamp}"

save_dir = os.path.join("models", "xgboost", "regression", unique_model_name)
os.makedirs(save_dir, exist_ok=True)

# Save the model (native XGBoost format)
model_path = os.path.join(save_dir, "model.json")
xgboost_reg_no_te.save_model(model_path)

# Save metadata (params + columns)
meta = {
    "model_name": unique_model_name,
    "created_at": timestamp,
    "model_type": "XGBRegressor",
    "task": "regression",
    "no_target_encoding": True,
    "target_range_clip": [0, 10],
    "hyperparameters": xgb_reg_params_no_te,
    "feature_count": int(X_train_xgb_reg_no_te.shape[1]),
    "feature_columns": X_train_xgb_reg_no_te.columns.astype(str).tolist(),
}

meta_path = os.path.join(save_dir, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"✅ Model saved to: {save_dir}")
print(f"   - {model_path}")
print(f"   - {meta_path}")

# -----------------------------
# 4) Predict + clip 0..10
# -----------------------------
pred_train_no_te = np.clip(xgboost_reg_no_te.predict(X_train_xgb_reg_no_te), 0, 10)
pred_test_no_te  = np.clip(xgboost_reg_no_te.predict(X_test_xgb_reg_no_te),  0, 10)

# -----------------------------
# 5) Metrics (train + test)
# -----------------------------
tol = 1.0

def regression_report(y_true, y_pred, tol=1.0):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))
    acc_tol = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100.0)
    return mae, rmse, r2, medae, mxerr, evs, acc_tol

(mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, evs_tr, acc_tr) = regression_report(y_train_xgb_reg_no_te, pred_train_no_te, tol=tol)
(mae_te, rmse_te, r2_te, medae_te, mxerr_te, evs_te, acc_te) = regression_report(y_test_xgb_reg_no_te,  pred_test_no_te,  tol=tol)

metrics_xgb_reg_no_te = pd.DataFrame({
    "Métrique": ["MAE", "RMSE", "R²", "MedianAE", "MaxError", "ExplainedVariance", f"Accuracy@±{tol}"],
    "Train": [mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, evs_tr, acc_tr],
    "Test":  [mae_te, rmse_te, r2_te, medae_te, mxerr_te, evs_te, acc_te],
    "Meilleur si": ["Plus bas", "Plus bas", "Plus haut", "Plus bas", "Plus bas", "Plus haut", "Plus haut"]
})

print("\n=== Résultats XGBoost Regressor (no_te) ===")
display(metrics_xgb_reg_no_te)

# -----------------------------
# 6) Plots (French titles)
# -----------------------------
plt.figure(figsize=(6,5))
plt.scatter(y_test_xgb_reg_no_te, pred_test_no_te, alpha=0.4)
mn = float(min(y_test_xgb_reg_no_te.min(), pred_test_no_te.min()))
mx = float(max(y_test_xgb_reg_no_te.max(), pred_test_no_te.max()))
plt.plot([mn, mx], [mn, mx], linestyle="--")
plt.xlabel("Note réelle")
plt.ylabel("Note prédite")
plt.title("Réel vs Prédit (jeu de test)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

resid = y_test_xgb_reg_no_te.values - pred_test_no_te
plt.figure(figsize=(6,4))
plt.hist(resid, bins=30, alpha=0.8)
plt.xlabel("Résidu (réel - prédit)")
plt.ylabel("Fréquence")
plt.title("Distribution des résidus (jeu de test)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

imp_df = pd.DataFrame({
    "variable": X_train_xgb_reg_no_te.columns.astype(str),
    "importance": xgboost_reg_no_te.feature_importances_
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(8,6))
plt.barh(imp_df["variable"][::-1], imp_df["importance"][::-1])
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.title("Top 20 variables importantes (XGBoost)")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

tols = np.arange(0.0, 2.01, 0.1)
acc_curve = [(np.abs(y_test_xgb_reg_no_te.values - pred_test_no_te) <= t).mean()*100 for t in tols]

plt.figure(figsize=(7,4))
plt.plot(tols, acc_curve, marker="o", linewidth=2)
plt.xlabel("Tolérance (points de note)")
plt.ylabel("Accuracy (%)")
plt.title("Courbe Accuracy selon la tolérance (jeu de test)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("""
Note importante :
- ROC-AUC est une métrique de classification, pas de régression.
- Pour une régression (score continu), on utilise MAE / RMSE / R² / Accuracy@±tol.
""")

```
