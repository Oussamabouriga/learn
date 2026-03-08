```
# ============================================================
# CATBOOST REGRESSION — BASELINE WITH TARGET WEIGHTS (sample_weight)
# Uses:
#   - train_pool_catboost_weighted
#   - test_pool_catboost_weighted
# Also uses:
#   - X_train_cb, X_test_cb
#   - y_train_no_te, y_test_no_te
#   - X_new_cb (example row already prepared)
# Same process as before:
#   - train
#   - metrics (MAE, RMSE, R2, MedianAE, MaxError, Accuracy@±tol)
#   - predict example
#   - SHAP global + SHAP local (example)
#   - save model
#
# Requirements:
#   pip install catboost shap
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error
)
import shap


# ==============================
# 0) Safety checks
# ==============================
print("Train pool rows:", train_pool_catboost_weighted.num_row())
print("Test  pool rows:", test_pool_catboost_weighted.num_row())

print("X_train_cb:", X_train_cb.shape, "| X_test_cb:", X_test_cb.shape)
print("y_train:", pd.Series(y_train_no_te).shape, "| y_test:", pd.Series(y_test_no_te).shape)

if "X_new_cb" in globals():
    print("X_new_cb:", X_new_cb.shape)
else:
    print("⚠️ X_new_cb not found. Create your example row dataframe first.")


# ==============================
# 1) Hyperparameters (editable dict)
# ==============================
cat_params_weighted = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",

    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,

    "random_strength": 1.0,
    "bagging_temperature": 1.0,

    "early_stopping_rounds": 200,

    "random_seed": 42,
    "verbose": 200,
    "allow_writing_files": False
}

model_name = "catboost_reg_baseline_weighted_no_te_v1"


# ==============================
# 2) Train weighted baseline model
# ==============================
cat_reg_weighted = CatBoostRegressor(**cat_params_weighted)

cat_reg_weighted.fit(
    train_pool_catboost_weighted,
    eval_set=test_pool_catboost_weighted,
    use_best_model=True
)

print("Model trained:", model_name)


# ==============================
# 3) Predictions (train/test) + clip 0..10
# ==============================
y_train_cb = pd.to_numeric(pd.Series(y_train_no_te), errors="coerce").astype(float).values
y_test_cb  = pd.to_numeric(pd.Series(y_test_no_te),  errors="coerce").astype(float).values

pred_train_cb_w = np.clip(cat_reg_weighted.predict(X_train_cb), 0, 10)
pred_test_cb_w  = np.clip(cat_reg_weighted.predict(X_test_cb),  0, 10)


# ==============================
# 4) Metrics + Accuracy@±tol
# ==============================
tol = 1.0  # set 0.5 if needed

mae_train = float(mean_absolute_error(y_train_cb, pred_train_cb_w))
rmse_train = float(np.sqrt(mean_squared_error(y_train_cb, pred_train_cb_w)))
r2_train = float(r2_score(y_train_cb, pred_train_cb_w))
medae_train = float(median_absolute_error(y_train_cb, pred_train_cb_w))
maxerr_train = float(max_error(y_train_cb, pred_train_cb_w))
acc_train = float((np.abs(y_train_cb - pred_train_cb_w) <= tol).mean() * 100)

mae_test = float(mean_absolute_error(y_test_cb, pred_test_cb_w))
rmse_test = float(np.sqrt(mean_squared_error(y_test_cb, pred_test_cb_w)))
r2_test = float(r2_score(y_test_cb, pred_test_cb_w))
medae_test = float(median_absolute_error(y_test_cb, pred_test_cb_w))
maxerr_test = float(max_error(y_test_cb, pred_test_cb_w))
acc_test = float((np.abs(y_test_cb - pred_test_cb_w) <= tol).mean() * 100)

metrics_cat_weighted = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", f"Accuracy@±{tol}"],
    "Train":  [mae_train, rmse_train, r2_train, medae_train, maxerr_train, acc_train],
    "Test":   [mae_test,  rmse_test,  r2_test,  medae_test,  maxerr_test,  acc_test],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher"]
})

print("\n=== CATBOOST REGRESSION — WEIGHTED BASELINE ===")
display(metrics_cat_weighted)


# ==============================
# 5) Predict your example row (X_new_cb)
# ==============================
if "X_new_cb" in globals():
    pred_example_w = float(np.clip(cat_reg_weighted.predict(X_new_cb)[0], 0, 10))
    print("\nExample prediction (CatBoost weighted):", pred_example_w)
else:
    pred_example_w = None


# ==============================
# 6) SHAP (global + example)
# ==============================
sample_size = min(300, len(X_test_cb))
X_shap = X_test_cb.sample(sample_size, random_state=42).copy()

explainer_cb_w = shap.TreeExplainer(cat_reg_weighted)
shap_values_cb_w = explainer_cb_w.shap_values(X_shap)

print("\nSHAP global (summary)")
shap.summary_plot(shap_values_cb_w, X_shap, show=True)
shap.summary_plot(shap_values_cb_w, X_shap, plot_type="bar", show=True)

if "X_new_cb" in globals():
    shap_one = explainer_cb_w.shap_values(X_new_cb)
    base_val = explainer_cb_w.expected_value

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

cat_reg_weighted.save_model(os.path.join(save_dir, "model.cbm"))
metrics_cat_weighted.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(cat_params_weighted, f, indent=2)

meta = {
    "model_name": model_name,
    "tol": tol,
    "example_prediction": pred_example_w
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved to:", save_dir)

```
