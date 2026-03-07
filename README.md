```
# ============================================================
# XGBoost Regressor — WEIGHTED + RANDOM SEARCH (no target encoding)
# + SAVE best model to: models/xgboost/regression/<model_name>/
#   - model.json
#   - metadata.json
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Base hyperparameters
# ==============================
xgb_reg_base_params_rs_w = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
    "missing": np.nan,
}

# ==============================
# 2) Random Search space
# ==============================
param_distributions_rs_w = {
    "n_estimators": [200, 400, 600, 900, 1200],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "max_depth": [3, 4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 5, 8, 12],
    "gamma": [0.0, 0.1, 0.3, 0.7, 1.0],
    "subsample": [0.6, 0.75, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.75, 0.8, 0.9, 1.0],
    "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 4.0, 8.0],
}

# ==============================
# 3) CV + search config
# ==============================
cv_rs_w = KFold(n_splits=5, shuffle=True, random_state=42)
n_iter_rs_w = 40
scoring_rs_w = "neg_root_mean_squared_error"

# ==============================
# 4) Build model + RandomizedSearchCV
# ==============================
xgb_reg_rs_w = XGBRegressor(**xgb_reg_base_params_rs_w)

rand_search_xgb_reg_w = RandomizedSearchCV(
    estimator=xgb_reg_rs_w,
    param_distributions=param_distributions_rs_w,
    n_iter=n_iter_rs_w,
    scoring=scoring_rs_w,
    cv=cv_rs_w,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    refit=True
)

# ==============================
# 5) Fit search WITH sample weights
# ==============================
rand_search_xgb_reg_w.fit(
    X_train_xgboost_reg_w,
    y_train_xgboost_reg_w,
    sample_weight=sample_weight_train_xgboost_reg_w
)

print("\n✅ Random Search done")
print("Best CV score (neg RMSE):", rand_search_xgb_reg_w.best_score_)
print("Best params:", rand_search_xgb_reg_w.best_params_)

# Best model already refit on full train
xgboost_reg_weighted_rs = rand_search_xgb_reg_w.best_estimator_

# ==============================
# 6) SAVE best model (unique folder)
# ==============================
timestamp = time.strftime("%Y%m%d_%H%M%S")
unique_model_name = f"xgb_reg_weighted_randomsearch_{timestamp}"

save_dir = os.path.join("models", "xgboost", "regression", unique_model_name)
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "model.json")
xgboost_reg_weighted_rs.save_model(model_path)

meta = {
    "model_name": unique_model_name,
    "created_at": timestamp,
    "model_type": "XGBRegressor",
    "task": "regression",
    "weighted_training": True,
    "weighting_method": "manual bins (0-6 / 7-8 / 9 / 10) + inverse freq + normalize + clip",
    "cv": {"type": "KFold", "n_splits": 5, "shuffle": True, "random_state": 42},
    "random_search": {"n_iter": n_iter_rs_w, "scoring": scoring_rs_w},
    "best_cv_score_neg_rmse": float(rand_search_xgb_reg_w.best_score_),
    "best_params": rand_search_xgb_reg_w.best_params_,
    "base_params": xgb_reg_base_params_rs_w,
    "feature_count": int(X_train_xgboost_reg_w.shape[1]),
    "feature_columns": X_train_xgboost_reg_w.columns.astype(str).tolist(),
}

meta_path = os.path.join(save_dir, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved to: {save_dir}")
print(f"   - {model_path}")
print(f"   - {meta_path}")

# ==============================
# 7) Predict (clip 0..10)
# ==============================
pred_train_rs = np.clip(xgboost_reg_weighted_rs.predict(X_train_xgboost_reg_w), 0, 10)
pred_test_rs  = np.clip(xgboost_reg_weighted_rs.predict(X_test_xgboost_reg_w),  0, 10)

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

mae_tr, rmse_tr, r2_tr, acc_tr = _metrics(y_train_xgboost_reg_w, pred_train_rs, tol=tol)
mae_te, rmse_te, r2_te, acc_te = _metrics(y_test_xgboost_reg_w,  pred_test_rs,  tol=tol)

metrics_xgb_weighted_rs = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train_weighted_RS": [mae_tr, rmse_tr, r2_tr, acc_tr],
    "Test_weighted_RS":  [mae_te, rmse_te, r2_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== XGBoost Weighted + Random Search Metrics ===")
display(metrics_xgb_weighted_rs)

# ==============================
# 9) Example prediction
# ==============================
if "X_new_xgboost_reg_w" in globals():
    X_one = X_new_xgboost_reg_w.copy()
else:
    X_one = X_new_encoded_no_te.copy()

X_one = X_one.reindex(columns=X_train_xgboost_reg_w.columns, fill_value=0).astype(float)

pred_example_rs = float(np.clip(xgboost_reg_weighted_rs.predict(X_one)[0], 0, 10))
print("\n✅ Prediction (example) — weighted + RS:", pred_example_rs)

# ==============================
# 10) SHAP Global + SHAP Local (example)
# ==============================
explainer_rs = shap.TreeExplainer(xgboost_reg_weighted_rs)

X_shap = X_test_xgboost_reg_w.copy()
sample_size = min(300, len(X_shap))
X_shap_sample = X_shap.sample(sample_size, random_state=42)

shap_values_global = explainer_rs.shap_values(X_shap_sample)

print("\n✅ SHAP global (summary)")
shap.summary_plot(shap_values_global, X_shap_sample, show=True)

print("\n✅ SHAP global (bar)")
shap.summary_plot(shap_values_global, X_shap_sample, plot_type="bar", show=True)

shap_values_one = explainer_rs.shap_values(X_one)
base_value = explainer_rs.expected_value
pred_raw = float(xgboost_reg_weighted_rs.predict(X_one)[0])

print("\n--- Lecture SHAP (local) ---")
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
