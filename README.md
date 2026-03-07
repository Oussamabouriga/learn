```
# ============================================================
# XGBoost Regressor — WEIGHTED + SMALL GRID SEARCH (no target encoding)
# - Uses your weighted dataset:
#     X_train_xgboost_reg_w, X_test_xgboost_reg_w
#     y_train_xgboost_reg_w, y_test_xgboost_reg_w
#     sample_weight_train_xgboost_reg_w
# - Small GridSearchCV (compute-friendly)
# - Train best model
# - Evaluate (MAE, RMSE, R2, Accuracy@±tol)
# - SHAP global + SHAP local (example)
# - Predict on your example row (X_new_xgboost_reg_w OR X_new_encoded_no_te)
# - Save model to: models/xgboost/regression/<model_name>/
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Base hyperparameters (fixed)
# ==============================
xgb_reg_base_params_gs_w = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
    "missing": np.nan,
}

# ==============================
# 2) Small Grid (focused)
# Keep it small to avoid heavy compute
# ==============================
param_grid_gs_w = {
    "n_estimators": [400, 700],
    "learning_rate": [0.03, 0.06],
    "max_depth": [5, 7],
    "min_child_weight": [3, 8],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [1.0, 2.0],
    # Keep gamma/alpha fixed (optional) to reduce grid size
}

# Total combinations = 2*2*2*2*2*2*2 = 128 (still ok).
# If you want even smaller, remove one or two parameters.

# ==============================
# 3) CV config
# ==============================
cv_gs_w = KFold(n_splits=5, shuffle=True, random_state=42)
scoring_gs_w = "neg_root_mean_squared_error"

# ==============================
# 4) Build estimator + GridSearchCV
# ==============================
xgb_reg_gs_w = XGBRegressor(**xgb_reg_base_params_gs_w)

grid_search_xgb_reg_w = GridSearchCV(
    estimator=xgb_reg_gs_w,
    param_grid=param_grid_gs_w,
    scoring=scoring_gs_w,
    cv=cv_gs_w,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# ==============================
# 5) Fit grid WITH sample weights
# ==============================
grid_search_xgb_reg_w.fit(
    X_train_xgboost_reg_w,
    y_train_xgboost_reg_w,
    sample_weight=sample_weight_train_xgboost_reg_w
)

print("\nGrid Search done")
print("Best CV score (neg RMSE):", grid_search_xgb_reg_w.best_score_)
print("Best params:", grid_search_xgb_reg_w.best_params_)

# Best model refit on full train
xgboost_reg_weighted_gs = grid_search_xgb_reg_w.best_estimator_

# ==============================
# 6) Save best model
# ==============================
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name_gs = f"xgb_reg_weighted_gridsearch_{timestamp}"
save_dir = os.path.join("models", "xgboost", "regression", model_name_gs)
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "model.json")
xgboost_reg_weighted_gs.save_model(model_path)

meta = {
    "model_name": model_name_gs,
    "created_at": timestamp,
    "model_type": "XGBRegressor",
    "task": "regression",
    "weighted_training": True,
    "weighting_method": "manual bins (0-6 / 7-8 / 9 / 10) + inverse freq + normalize + clip",
    "cv": {"type": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
    "grid_search": {"param_grid": param_grid_gs_w, "scoring": scoring_gs_w},
    "best_cv_score_neg_rmse": float(grid_search_xgb_reg_w.best_score_),
    "best_params": grid_search_xgb_reg_w.best_params_,
    "base_params": xgb_reg_base_params_gs_w,
    "feature_count": int(X_train_xgboost_reg_w.shape[1]),
    "feature_columns": X_train_xgboost_reg_w.columns.astype(str).tolist(),
}
meta_path = os.path.join(save_dir, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("Model saved in:", save_dir)

# ==============================
# 7) Predict (clip 0..10)
# ==============================
pred_train_gs = np.clip(xgboost_reg_weighted_gs.predict(X_train_xgboost_reg_w), 0, 10)
pred_test_gs  = np.clip(xgboost_reg_weighted_gs.predict(X_test_xgboost_reg_w),  0, 10)

# ==============================
# 8) Metrics (MAE, RMSE, R2, Accuracy@±tol)
# ==============================
tol = 1.0

def _metrics(y_true, y_pred, tol=1.0):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100)
    return mae, rmse, r2, acc

mae_tr, rmse_tr, r2_tr, acc_tr = _metrics(y_train_xgboost_reg_w, pred_train_gs, tol=tol)
mae_te, rmse_te, r2_te, acc_te = _metrics(y_test_xgboost_reg_w,  pred_test_gs,  tol=tol)

metrics_xgb_weighted_gs = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train_weighted_GS": [mae_tr, rmse_tr, r2_tr, acc_tr],
    "Test_weighted_GS":  [mae_te, rmse_te, r2_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\nXGBoost Weighted + Grid Search Metrics")
display(metrics_xgb_weighted_gs)

# ==============================
# 9) Example prediction
# ==============================
if "X_new_xgboost_reg_w" in globals():
    X_one = X_new_xgboost_reg_w.copy()
else:
    X_one = X_new_encoded_no_te.copy()

X_one = X_one.reindex(columns=X_train_xgboost_reg_w.columns, fill_value=0).astype(float)

pred_example_gs = float(np.clip(xgboost_reg_weighted_gs.predict(X_one)[0], 0, 10))
print("\nPrediction (example) — weighted + GS:", pred_example_gs)

# ==============================
# 10) SHAP Global + SHAP Local (example)
# ==============================
explainer_gs = shap.TreeExplainer(xgboost_reg_weighted_gs)

X_shap = X_test_xgboost_reg_w.copy()
sample_size = min(300, len(X_shap))
X_shap_sample = X_shap.sample(sample_size, random_state=42)

shap_values_global = explainer_gs.shap_values(X_shap_sample)

print("\nSHAP global (summary)")
shap.summary_plot(shap_values_global, X_shap_sample, show=True)

print("\nSHAP global (bar)")
shap.summary_plot(shap_values_global, X_shap_sample, plot_type="bar", show=True)

shap_values_one = explainer_gs.shap_values(X_one)
base_value = explainer_gs.expected_value
pred_raw = float(xgboost_reg_weighted_gs.predict(X_one)[0])

print("\nLecture SHAP (local)")
print("E[f(X)] (baseline) =", base_value)
print("f(X) (prediction raw) =", pred_raw)
print("Somme(SHAP) + baseline ≈ prediction")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[0],
        base_values=base_value,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()

```
