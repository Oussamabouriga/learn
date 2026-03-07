```

# ============================================================
# XGBoost Regressor — WEIGHTED TRAINING (Imbalanced Regression)
# + Hyperparameters dict (same style as before)
# + Train with sample_weight
# + Evaluate (MAE, RMSE, R2, Accuracy@±tol)
# + SHAP global + SHAP local (example)
#
# Assumes you already prepared:
#   - X_train_xgboost_reg_w, X_test_xgboost_reg_w
#   - y_train_xgboost_reg_w, y_test_xgboost_reg_w
#   - sample_weight_train_xgboost_reg_w
# And you have an example row encoded:
#   - X_new_xgboost_reg_w   OR   X_new_encoded_no_te
# ============================================================

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Hyperparameters (baseline, same as before)
# ==============================
xgb_reg_params_weighted = {
    # Regression objective
    "objective": "reg:squarederror",
    "eval_metric": "rmse",

    # Main hyperparameters (good baseline)
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "gamma": 0.0,

    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,

    "reg_alpha": 0.0,
    "reg_lambda": 1.0,

    "tree_method": "hist",
    "grow_policy": "depthwise",
    "max_leaves": 0,

    # Missing values handling
    "missing": np.nan,

    # Performance / reproducibility
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

print("✅ Hyperparameters (weighted model):")
display(pd.DataFrame([xgb_reg_params_weighted]).T.rename(columns={0: "value"}))


# ==============================
# 2) Safety checks (shapes / dtypes)
# ==============================
print("\nShapes:")
print("X_train:", X_train_xgboost_reg_w.shape)
print("X_test :", X_test_xgboost_reg_w.shape)
print("y_train:", y_train_xgboost_reg_w.shape)
print("y_test :", y_test_xgboost_reg_w.shape)
print("sample_weight_train:", sample_weight_train_xgboost_reg_w.shape)

print("\nAny non-numeric columns in X_train?")
non_num_cols = X_train_xgboost_reg_w.select_dtypes(exclude=[np.number]).columns.tolist()
print(non_num_cols)

# Ensure float (recommended for xgboost)
X_train_xgboost_reg_w = X_train_xgboost_reg_w.astype(float)
X_test_xgboost_reg_w  = X_test_xgboost_reg_w.astype(float)

y_train_xgboost_reg_w = pd.to_numeric(y_train_xgboost_reg_w, errors="coerce").astype(float)
y_test_xgboost_reg_w  = pd.to_numeric(y_test_xgboost_reg_w,  errors="coerce").astype(float)

# Drop missing targets if any
train_mask = y_train_xgboost_reg_w.notna()
test_mask  = y_test_xgboost_reg_w.notna()

X_train_xgboost_reg_w = X_train_xgboost_reg_w.loc[train_mask].copy()
y_train_xgboost_reg_w = y_train_xgboost_reg_w.loc[train_mask].copy()
sample_weight_train_xgboost_reg_w = np.asarray(sample_weight_train_xgboost_reg_w)[train_mask.values]

X_test_xgboost_reg_w = X_test_xgboost_reg_w.loc[test_mask].copy()
y_test_xgboost_reg_w = y_test_xgboost_reg_w.loc[test_mask].copy()

print("\n✅ After cleaning missing targets:")
print("X_train:", X_train_xgboost_reg_w.shape, "| y_train:", y_train_xgboost_reg_w.shape, "| weights:", sample_weight_train_xgboost_reg_w.shape)
print("X_test :", X_test_xgboost_reg_w.shape,  "| y_test :", y_test_xgboost_reg_w.shape)


# ==============================
# 3) Train WEIGHTED model
# ==============================
xgboost_reg_weighted = XGBRegressor(**xgb_reg_params_weighted)

xgboost_reg_weighted.fit(
    X_train_xgboost_reg_w,
    y_train_xgboost_reg_w,
    sample_weight=sample_weight_train_xgboost_reg_w
)

print("\n✅ xgboost_reg_weighted trained successfully")


# ==============================
# 4) Predict (clip to business range 0..10)
# ==============================
pred_train_w = np.clip(xgboost_reg_weighted.predict(X_train_xgboost_reg_w), 0, 10)
pred_test_w  = np.clip(xgboost_reg_weighted.predict(X_test_xgboost_reg_w),  0, 10)


# ==============================
# 5) Metrics + Accuracy@±tol
# ==============================
tol = 1.0  # change to 0.5 if you want

mae_train_w  = float(mean_absolute_error(y_train_xgboost_reg_w, pred_train_w))
rmse_train_w = float(np.sqrt(mean_squared_error(y_train_xgboost_reg_w, pred_train_w)))
r2_train_w   = float(r2_score(y_train_xgboost_reg_w, pred_train_w))
acc_train_w  = float((np.abs(y_train_xgboost_reg_w.values - pred_train_w) <= tol).mean() * 100)

mae_test_w   = float(mean_absolute_error(y_test_xgboost_reg_w, pred_test_w))
rmse_test_w  = float(np.sqrt(mean_squared_error(y_test_xgboost_reg_w, pred_test_w)))
r2_test_w    = float(r2_score(y_test_xgboost_reg_w, pred_test_w))
acc_test_w   = float((np.abs(y_test_xgboost_reg_w.values - pred_test_w) <= tol).mean() * 100)

metrics_xgb_weighted = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train_weighted": [mae_train_w, rmse_train_w, r2_train_w, acc_train_w],
    "Test_weighted":  [mae_test_w,  rmse_test_w,  r2_test_w,  acc_test_w],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== XGBoost Weighted Metrics ===")
display(metrics_xgb_weighted)


# ==============================
# 6) Feature importance (built-in)
# ==============================
feature_importance_xgb_weighted = pd.DataFrame({
    "feature": X_train_xgboost_reg_w.columns.astype(str),
    "importance": xgboost_reg_weighted.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 features (XGBoost weighted):")
display(feature_importance_xgb_weighted.head(20))


# ==============================
# 7) Predict your example row (encoded) + SHAP local
# ==============================
# Accept 2 cases:
# - If you already have X_new_xgboost_reg_w, use it
# - Else use X_new_encoded_no_te and align it to weighted columns

if "X_new_xgboost_reg_w" in globals():
    X_one_w = X_new_xgboost_reg_w.copy()
else:
    X_one_w = X_new_encoded_no_te.copy()

# align columns exactly like train
X_one_w = X_one_w.reindex(columns=X_train_xgboost_reg_w.columns, fill_value=0).astype(float)

print("\nX_one_w.shape =", X_one_w.shape)

pred_example_w = float(np.clip(xgboost_reg_weighted.predict(X_one_w)[0], 0, 10))
print("\n✅ Weighted model prediction (example) =", pred_example_w)


# ==============================
# 8) SHAP — Global + Local (example)
# ==============================
explainer_w = shap.TreeExplainer(xgboost_reg_weighted)

# ---- Global SHAP on test sample
X_shap_w = X_test_xgboost_reg_w.copy()
sample_size = min(300, len(X_shap_w))
X_shap_sample_w = X_shap_w.sample(sample_size, random_state=42)

shap_values_global_w = explainer_w.shap_values(X_shap_sample_w)

print("\n✅ SHAP global (summary)")
shap.summary_plot(shap_values_global_w, X_shap_sample_w, show=True)

print("\n✅ SHAP global (bar)")
shap.summary_plot(shap_values_global_w, X_shap_sample_w, plot_type="bar", show=True)

# ---- Local SHAP for example
shap_values_one_w = explainer_w.shap_values(X_one_w)
base_value_w = explainer_w.expected_value
pred_raw_w = float(xgboost_reg_weighted.predict(X_one_w)[0])

print("\n--- Lecture SHAP (local) ---")
print("E[f(X)] (baseline) =", base_value_w)
print("f(X) (prediction raw) =", pred_raw_w)
print("Somme(SHAP) + baseline ≈ prediction")

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one_w[0],
        base_values=base_value_w,
        data=X_one_w.iloc[0],
        feature_names=X_one_w.columns
    ),
    max_display=20
)
plt.show()
```
