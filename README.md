```

# ============================================================
# CATBOOST REGRESSION — BASELINE (NO target weights)
# Same style as your XGBoost baseline:
# - uses Pools already prepared (train_pool_catboost_baseline / test_pool_catboost_baseline)
# - trains CatBoostRegressor
# - evaluates MAE / RMSE / R2 / MedianAE / MaxError / Accuracy@±tol
# - predicts your example row (X_new_cb) + local explanation
# - SHAP global + SHAP local (example)
# - saves model to: models/catboost/regression/<model_name>/
#
# REQUIREMENTS:
#   pip install catboost shap joblib
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error
)
import shap


# ==============================
# 0) Safety checks (Pools + example row)
# ==============================
# Expecting you already have these variables from your previous cells:
# - train_pool_catboost_baseline
# - test_pool_catboost_baseline
# - X_train_cb, X_test_cb
# - y_train_no_te, y_test_no_te
# - cat_cols_cb (categorical columns list)
# - X_new_cb (example row prepared with same transformations)

print("Pool train:", train_pool_catboost_baseline.num_row(), "rows")
print("Pool test :", test_pool_catboost_baseline.num_row(), "rows")

print("X_train_cb:", X_train_cb.shape, "| X_test_cb:", X_test_cb.shape)
print("y_train:", pd.Series(y_train_no_te).shape, "| y_test:", pd.Series(y_test_no_te).shape)

if "X_new_cb" in globals():
    print("X_new_cb:", X_new_cb.shape)
else:
    print("⚠️ X_new_cb not found. Create your example row dataframe first.")


# ==============================
# 1) Hyperparameters (editable dict)
# ==============================
cat_params_baseline = {
    # Loss / objective
    "loss_function": "RMSE",     # regression
    "eval_metric": "RMSE",

    # Core training
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,

    # Regularization / randomness
    "random_strength": 1.0,
    "bagging_temperature": 1.0,

    # Overfitting control
    "early_stopping_rounds": 200,

    # Misc
    "random_seed": 42,
    "verbose": 200,
    "allow_writing_files": False
}

model_name = "catboost_reg_baseline_no_te_v1"


# ==============================
# 2) Train baseline CatBoost regressor (NO weights)
# ==============================
cat_reg_baseline = CatBoostRegressor(**cat_params_baseline)

cat_reg_baseline.fit(
    train_pool_catboost_baseline,
    eval_set=test_pool_catboost_baseline,
    use_best_model=True
)

print("Model trained:", model_name)


# ==============================
# 3) Predictions (train/test) + clip to 0..10
# ==============================
y_train_cb = pd.to_numeric(pd.Series(y_train_no_te), errors="coerce").astype(float).values
y_test_cb  = pd.to_numeric(pd.Series(y_test_no_te),  errors="coerce").astype(float).values

pred_train_cb = np.clip(cat_reg_baseline.predict(X_train_cb), 0, 10)
pred_test_cb  = np.clip(cat_reg_baseline.predict(X_test_cb),  0, 10)


# ==============================
# 4) Metrics + Accuracy@±tol
# ==============================
tol = 1.0  # change to 0.5 if you want

mae_train = float(mean_absolute_error(y_train_cb, pred_train_cb))
rmse_train = float(np.sqrt(mean_squared_error(y_train_cb, pred_train_cb)))
r2_train = float(r2_score(y_train_cb, pred_train_cb))
medae_train = float(median_absolute_error(y_train_cb, pred_train_cb))
maxerr_train = float(max_error(y_train_cb, pred_train_cb))
acc_train = float((np.abs(y_train_cb - pred_train_cb) <= tol).mean() * 100)

mae_test = float(mean_absolute_error(y_test_cb, pred_test_cb))
rmse_test = float(np.sqrt(mean_squared_error(y_test_cb, pred_test_cb)))
r2_test = float(r2_score(y_test_cb, pred_test_cb))
medae_test = float(median_absolute_error(y_test_cb, pred_test_cb))
maxerr_test = float(max_error(y_test_cb, pred_test_cb))
acc_test = float((np.abs(y_test_cb - pred_test_cb) <= tol).mean() * 100)

metrics_cat_baseline = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", f"Accuracy@±{tol}"],
    "Train":  [mae_train, rmse_train, r2_train, medae_train, maxerr_train, acc_train],
    "Test":   [mae_test,  rmse_test,  r2_test,  medae_test,  maxerr_test,  acc_test],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher"]
})

print("\n=== CATBOOST REGRESSION — BASELINE (NO weights) ===")
display(metrics_cat_baseline)


# ==============================
# 5) Predict your example row (X_new_cb)
# ==============================
if "X_new_cb" in globals():
    pred_example = float(np.clip(cat_reg_baseline.predict(X_new_cb)[0], 0, 10))
    print("\nExample prediction (CatBoost baseline):", pred_example)
else:
    pred_example = None


# ==============================
# 6) SHAP (global + example)
# Notes:
# - For CatBoost, TreeExplainer works well.
# - Use a small sample for speed.
# ==============================
# SHAP sample (from test)
sample_size = min(300, len(X_test_cb))
X_shap = X_test_cb.sample(sample_size, random_state=42).copy()

explainer_cb = shap.TreeExplainer(cat_reg_baseline)
shap_values_cb = explainer_cb.shap_values(X_shap)

print("\nSHAP global (summary)")
shap.summary_plot(shap_values_cb, X_shap, show=True)
shap.summary_plot(shap_values_cb, X_shap, plot_type="bar", show=True)

# Local SHAP for example
if "X_new_cb" in globals():
    shap_one = explainer_cb.shap_values(X_new_cb)
    base_val = explainer_cb.expected_value

    # Waterfall: keep it readable
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_one[0],
            base_values=base_val,
            data=X_new_cb.iloc[0],
            feature_names=X_new_cb.columns
        ),
        max_display=20
    )
    plt.show()


# ==============================
# 7) Save model + metadata
# ==============================
save_dir = os.path.join("models", "catboost", "regression", model_name)
os.makedirs(save_dir, exist_ok=True)

# CatBoost has its own format (recommended)
cat_reg_baseline.save_model(os.path.join(save_dir, "model.cbm"))

# Also save metrics + params
metrics_cat_baseline.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(cat_params_baseline, f, indent=2)

meta = {
    "model_name": model_name,
    "tol": tol,
    "example_prediction": pred_example
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved to:", save_dir)
```
