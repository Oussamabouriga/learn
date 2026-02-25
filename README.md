```
# ============================================================
# XGBOOST REGRESSION (IMBALANCED TARGET) + RANDOM SEARCH + SMALL GRID SEARCH
# FULL NOTEBOOK-STYLE CODE (NO FUNCTIONS)
# ============================================================
# Assumes you ALREADY have:
#   - X_train_encoded, X_test_encoded
#   - y_train, y_test
#   - X_new_encoded  (your example row already encoded with same training encoding)
#
# This code will:
#   1) Build regression sample weights for imbalanced target (train only)
#   2) Train a weighted baseline XGBoost model
#   3) Run Random Search (weighted)
#   4) Run Small Grid Search around Random Search best params (weighted)
#   5) Evaluate all models (metrics + regression "accuracy")
#   6) SHAP global importance (best model)
#   7) SHAP local explanation for your example row
#   8) Predict your example row with both tuned models (Random + Grid)
# ============================================================


# ============================================================
# 1) Imports
# ============================================================
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Optional display settings for notebooks
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)


# ============================================================
# 2) Final checks + numeric safety
# ============================================================
print("X_train_encoded shape:", X_train_encoded.shape)
print("X_test_encoded shape :", X_test_encoded.shape)
print("y_train shape        :", y_train.shape)
print("y_test shape         :", y_test.shape)

# Convert features to numeric float
for c in X_train_encoded.columns:
    X_train_encoded[c] = pd.to_numeric(X_train_encoded[c], errors="coerce")
for c in X_test_encoded.columns:
    X_test_encoded[c] = pd.to_numeric(X_test_encoded[c], errors="coerce")

X_train_encoded = X_train_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)

# Convert target to numeric float
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test = pd.to_numeric(y_test, errors="coerce").astype(float)

# Align / clean X_new_encoded if present
if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).copy()
    for c in X_new_encoded.columns:
        X_new_encoded[c] = pd.to_numeric(X_new_encoded[c], errors="coerce")
    X_new_encoded = X_new_encoded.astype(float)
    print("X_new_encoded shape  :", X_new_encoded.shape)

print("\nAny non-numeric columns left in X_train_encoded?")
print(X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist())

print("\nDone ✅")


# ============================================================
# 3) Build regression sample weights for imbalanced target (TRAIN ONLY)
#    Method: inverse frequency on target bins (quantile bins)
# ============================================================
# If your target has many repeated values (e.g., many 10s), qcut with duplicates='drop' is important
n_target_bins_for_weights = 10

# Bin train target
y_train_bins = pd.qcut(y_train, q=n_target_bins_for_weights, duplicates="drop")

# Frequency counts per bin
target_bin_counts = y_train_bins.value_counts(dropna=False).sort_index()

# Inverse frequency map
target_bin_weight_map = (1.0 / target_bin_counts).to_dict()

# Row-level weights
sample_weight_train = y_train_bins.map(target_bin_weight_map).astype(float)

# Normalize weights so average ≈ 1 (stability)
sample_weight_train = sample_weight_train / sample_weight_train.mean()

# Optional: clip extreme weights (helps prevent instability/overfitting on rare bins)
sample_weight_train = np.clip(sample_weight_train, 0.5, 5.0)
sample_weight_train = sample_weight_train / sample_weight_train.mean()

print("Sample weights ready ✅")
print("Weight summary:")
print(pd.Series(sample_weight_train).describe())

print("\nTarget bin distribution (train):")
display(target_bin_counts)

print("\nAverage weight by target bin:")
weights_debug_df = pd.DataFrame({
    "y_train": y_train.values,
    "y_bin": y_train_bins.astype(str).values,
    "weight": np.asarray(sample_weight_train, dtype=float)
})
display(
    weights_debug_df.groupby("y_bin")["weight"]
    .agg(["count", "mean", "min", "max"])
    .sort_index()
)


# ============================================================
# 4) Common CV setup (used by Random Search and Grid Search)
# ============================================================
cv_kfold_weighted = KFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 5) Model 1: Weighted Baseline XGBoost (manual params)
#    (kept for comparison vs Random Search / Grid Search)
# ============================================================
model_xgb_weighted_baseline = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",

    # Baseline params
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=3,
    gamma=0.0,

    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=1.0,
    colsample_bynode=1.0,

    reg_alpha=0.0,
    reg_lambda=1.0,

    tree_method="hist",
    grow_policy="depthwise",
    max_leaves=0,

    missing=np.nan,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

model_xgb_weighted_baseline.fit(
    X_train_encoded,
    y_train,
    sample_weight=np.asarray(sample_weight_train, dtype=float)
)

print("Model 1 trained: model_xgb_weighted_baseline ✅")


# ============================================================
# 6) Evaluate Model 1 (Weighted Baseline)
# ============================================================
pred_train_baseline_w = model_xgb_weighted_baseline.predict(X_train_encoded)
pred_test_baseline_w = model_xgb_weighted_baseline.predict(X_test_encoded)

# Clip to business range (0..10) if target is score/note
pred_train_baseline_w_clip = np.clip(pred_train_baseline_w, 0, 10)
pred_test_baseline_w_clip = np.clip(pred_test_baseline_w, 0, 10)

# Metrics
mae_train_baseline_w = mean_absolute_error(y_train, pred_train_baseline_w_clip)
mse_train_baseline_w = mean_squared_error(y_train, pred_train_baseline_w_clip)
rmse_train_baseline_w = np.sqrt(mse_train_baseline_w)
r2_train_baseline_w = r2_score(y_train, pred_train_baseline_w_clip)
medae_train_baseline_w = median_absolute_error(y_train, pred_train_baseline_w_clip)
maxerr_train_baseline_w = max_error(y_train, pred_train_baseline_w_clip)
evs_train_baseline_w = explained_variance_score(y_train, pred_train_baseline_w_clip)

mae_test_baseline_w = mean_absolute_error(y_test, pred_test_baseline_w_clip)
mse_test_baseline_w = mean_squared_error(y_test, pred_test_baseline_w_clip)
rmse_test_baseline_w = np.sqrt(mse_test_baseline_w)
r2_test_baseline_w = r2_score(y_test, pred_test_baseline_w_clip)
medae_test_baseline_w = median_absolute_error(y_test, pred_test_baseline_w_clip)
maxerr_test_baseline_w = max_error(y_test, pred_test_baseline_w_clip)
evs_test_baseline_w = explained_variance_score(y_test, pred_test_baseline_w_clip)

# Regression "accuracy" (tolerance-based)
acc05_train_baseline_w = (np.abs(y_train - pred_train_baseline_w_clip) <= 0.5).mean() * 100
acc10_train_baseline_w = (np.abs(y_train - pred_train_baseline_w_clip) <= 1.0).mean() * 100
acc05_test_baseline_w = (np.abs(y_test - pred_test_baseline_w_clip) <= 0.5).mean() * 100
acc10_test_baseline_w = (np.abs(y_test - pred_test_baseline_w_clip) <= 1.0).mean() * 100

eps = 1e-8
mape_test_baseline_w = np.mean(np.abs((y_test - pred_test_baseline_w_clip) / np.maximum(np.abs(y_test), eps))) * 100
smape_test_baseline_w = np.mean(
    2.0 * np.abs(pred_test_baseline_w_clip - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_baseline_w_clip), eps)
) * 100


# ============================================================
# 7) Model 2: Random Search (weighted)
# ============================================================
# Base estimator for random search
xgb_random_base_weighted = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    missing=np.nan
)

# Focused parameter space (good first search, not too huge)
param_distributions_random_weighted = {
    "n_estimators": [200, 300, 500, 700, 900],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.8, 1.0],
    "colsample_bynode": [0.8, 1.0],
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "grow_policy": ["depthwise", "lossguide"],
    "max_leaves": [0, 31, 63, 127]
}

random_search_xgb_weighted = RandomizedSearchCV(
    estimator=xgb_random_base_weighted,
    param_distributions=param_distributions_random_weighted,
    n_iter=30,   # adjust 20-50 depending on compute budget
    scoring="neg_root_mean_squared_error",
    cv=cv_kfold_weighted,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

random_search_xgb_weighted.fit(
    X_train_encoded,
    y_train,
    sample_weight=np.asarray(sample_weight_train, dtype=float)
)

model_xgb_weighted_randomsearch = random_search_xgb_weighted.best_estimator_

print("\nModel 2 trained: model_xgb_weighted_randomsearch ✅")
print("Best Random Search CV RMSE:", -random_search_xgb_weighted.best_score_)
print("Best Random Search Params:")
print(random_search_xgb_weighted.best_params_)


# ============================================================
# 8) Evaluate Model 2 (Random Search weighted)
# ============================================================
pred_train_random_w = model_xgb_weighted_randomsearch.predict(X_train_encoded)
pred_test_random_w = model_xgb_weighted_randomsearch.predict(X_test_encoded)

pred_train_random_w_clip = np.clip(pred_train_random_w, 0, 10)
pred_test_random_w_clip = np.clip(pred_test_random_w, 0, 10)

mae_train_random_w = mean_absolute_error(y_train, pred_train_random_w_clip)
mse_train_random_w = mean_squared_error(y_train, pred_train_random_w_clip)
rmse_train_random_w = np.sqrt(mse_train_random_w)
r2_train_random_w = r2_score(y_train, pred_train_random_w_clip)
medae_train_random_w = median_absolute_error(y_train, pred_train_random_w_clip)
maxerr_train_random_w = max_error(y_train, pred_train_random_w_clip)
evs_train_random_w = explained_variance_score(y_train, pred_train_random_w_clip)

mae_test_random_w = mean_absolute_error(y_test, pred_test_random_w_clip)
mse_test_random_w = mean_squared_error(y_test, pred_test_random_w_clip)
rmse_test_random_w = np.sqrt(mse_test_random_w)
r2_test_random_w = r2_score(y_test, pred_test_random_w_clip)
medae_test_random_w = median_absolute_error(y_test, pred_test_random_w_clip)
maxerr_test_random_w = max_error(y_test, pred_test_random_w_clip)
evs_test_random_w = explained_variance_score(y_test, pred_test_random_w_clip)

acc05_train_random_w = (np.abs(y_train - pred_train_random_w_clip) <= 0.5).mean() * 100
acc10_train_random_w = (np.abs(y_train - pred_train_random_w_clip) <= 1.0).mean() * 100
acc05_test_random_w = (np.abs(y_test - pred_test_random_w_clip) <= 0.5).mean() * 100
acc10_test_random_w = (np.abs(y_test - pred_test_random_w_clip) <= 1.0).mean() * 100

mape_test_random_w = np.mean(np.abs((y_test - pred_test_random_w_clip) / np.maximum(np.abs(y_test), eps))) * 100
smape_test_random_w = np.mean(
    2.0 * np.abs(pred_test_random_w_clip - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_random_w_clip), eps)
) * 100


# ============================================================
# 9) Inspect Random Search CV results (top trials)
# ============================================================
cv_results_random_weighted_df = (
    pd.DataFrame(random_search_xgb_weighted.cv_results_)
    .sort_values("rank_test_score")
    .reset_index(drop=True)
)

cols_show_random = [
    "rank_test_score", "mean_test_score", "std_test_score", "mean_train_score",
    "param_n_estimators", "param_learning_rate", "param_max_depth", "param_min_child_weight",
    "param_gamma", "param_subsample", "param_colsample_bytree",
    "param_reg_alpha", "param_reg_lambda", "param_grow_policy", "param_max_leaves"
]
print("\nTop Random Search trials:")
display(cv_results_random_weighted_df[[c for c in cols_show_random if c in cv_results_random_weighted_df.columns]].head(10))


# ============================================================
# 10) Model 3: Small Grid Search (weighted) around Random Search best params
# ============================================================
best_rs_params = random_search_xgb_weighted.best_params_.copy()

# Small focused grid around best params (cheap refinement)
# We tune only the most impactful params (keep others fixed)
base_lr = float(best_rs_params.get("learning_rate", 0.05))
base_depth = int(best_rs_params.get("max_depth", 6))
base_min_child = int(best_rs_params.get("min_child_weight", 3))
base_subsample = float(best_rs_params.get("subsample", 0.8))
base_colsample = float(best_rs_params.get("colsample_bytree", 0.8))
base_n_estimators = int(best_rs_params.get("n_estimators", 500))
base_gamma = float(best_rs_params.get("gamma", 0.0))
base_reg_alpha = float(best_rs_params.get("reg_alpha", 0.0))
base_reg_lambda = float(best_rs_params.get("reg_lambda", 1.0))
base_grow_policy = best_rs_params.get("grow_policy", "depthwise")
base_max_leaves = int(best_rs_params.get("max_leaves", 0))
base_colsample_bylevel = float(best_rs_params.get("colsample_bylevel", 1.0))
base_colsample_bynode = float(best_rs_params.get("colsample_bynode", 1.0))

# Helper values (manual inline, no function)
lr_grid = sorted(set([
    round(max(0.005, base_lr * 0.7), 4),
    round(base_lr, 4),
    round(min(0.3, base_lr * 1.3), 4)
]))

depth_grid = sorted(set([
    max(2, base_depth - 1),
    base_depth,
    min(12, base_depth + 1)
]))

min_child_grid = sorted(set([
    max(1, base_min_child - 2),
    base_min_child,
    base_min_child + 2
]))

subsample_grid = sorted(set([
    round(max(0.5, base_subsample - 0.1), 2),
    round(base_subsample, 2),
    round(min(1.0, base_subsample + 0.1), 2)
]))

colsample_grid = sorted(set([
    round(max(0.5, base_colsample - 0.1), 2),
    round(base_colsample, 2),
    round(min(1.0, base_colsample + 0.1), 2)
]))

# Keep grid small to avoid huge compute
param_grid_small_weighted = {
    "n_estimators": [base_n_estimators],  # keep fixed to reduce compute
    "learning_rate": lr_grid,
    "max_depth": depth_grid,
    "min_child_weight": min_child_grid,
    "subsample": subsample_grid,
    "colsample_bytree": colsample_grid,

    # Fixed from random search best
    "gamma": [base_gamma],
    "reg_alpha": [base_reg_alpha],
    "reg_lambda": [base_reg_lambda],
    "grow_policy": [base_grow_policy],
    "max_leaves": [base_max_leaves],
    "colsample_bylevel": [base_colsample_bylevel],
    "colsample_bynode": [base_colsample_bynode]
}

print("\nSmall Grid (weighted) around Random Search best:")
print(param_grid_small_weighted)

xgb_grid_base_weighted = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    missing=np.nan
)

grid_search_xgb_weighted_small = GridSearchCV(
    estimator=xgb_grid_base_weighted,
    param_grid=param_grid_small_weighted,
    scoring="neg_root_mean_squared_error",
    cv=cv_kfold_weighted,
    verbose=2,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

grid_search_xgb_weighted_small.fit(
    X_train_encoded,
    y_train,
    sample_weight=np.asarray(sample_weight_train, dtype=float)
)

model_xgb_weighted_gridsearch_small = grid_search_xgb_weighted_small.best_estimator_

print("\nModel 3 trained: model_xgb_weighted_gridsearch_small ✅")
print("Best Small Grid Search CV RMSE:", -grid_search_xgb_weighted_small.best_score_)
print("Best Small Grid Search Params:")
print(grid_search_xgb_weighted_small.best_params_)


# ============================================================
# 11) Evaluate Model 3 (Small Grid Search weighted)
# ============================================================
pred_train_grid_w = model_xgb_weighted_gridsearch_small.predict(X_train_encoded)
pred_test_grid_w = model_xgb_weighted_gridsearch_small.predict(X_test_encoded)

pred_train_grid_w_clip = np.clip(pred_train_grid_w, 0, 10)
pred_test_grid_w_clip = np.clip(pred_test_grid_w, 0, 10)

mae_train_grid_w = mean_absolute_error(y_train, pred_train_grid_w_clip)
mse_train_grid_w = mean_squared_error(y_train, pred_train_grid_w_clip)
rmse_train_grid_w = np.sqrt(mse_train_grid_w)
r2_train_grid_w = r2_score(y_train, pred_train_grid_w_clip)
medae_train_grid_w = median_absolute_error(y_train, pred_train_grid_w_clip)
maxerr_train_grid_w = max_error(y_train, pred_train_grid_w_clip)
evs_train_grid_w = explained_variance_score(y_train, pred_train_grid_w_clip)

mae_test_grid_w = mean_absolute_error(y_test, pred_test_grid_w_clip)
mse_test_grid_w = mean_squared_error(y_test, pred_test_grid_w_clip)
rmse_test_grid_w = np.sqrt(mse_test_grid_w)
r2_test_grid_w = r2_score(y_test, pred_test_grid_w_clip)
medae_test_grid_w = median_absolute_error(y_test, pred_test_grid_w_clip)
maxerr_test_grid_w = max_error(y_test, pred_test_grid_w_clip)
evs_test_grid_w = explained_variance_score(y_test, pred_test_grid_w_clip)

acc05_train_grid_w = (np.abs(y_train - pred_train_grid_w_clip) <= 0.5).mean() * 100
acc10_train_grid_w = (np.abs(y_train - pred_train_grid_w_clip) <= 1.0).mean() * 100
acc05_test_grid_w = (np.abs(y_test - pred_test_grid_w_clip) <= 0.5).mean() * 100
acc10_test_grid_w = (np.abs(y_test - pred_test_grid_w_clip) <= 1.0).mean() * 100

mape_test_grid_w = np.mean(np.abs((y_test - pred_test_grid_w_clip) / np.maximum(np.abs(y_test), eps))) * 100
smape_test_grid_w = np.mean(
    2.0 * np.abs(pred_test_grid_w_clip - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_grid_w_clip), eps)
) * 100


# ============================================================
# 12) Metrics tables for all three models
# ============================================================
metrics_all_models_test = pd.DataFrame({
    "Metric": [
        "MAE", "MSE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance",
        "Accuracy@±0.5 (%)", "Accuracy@±1.0 (%)", "MAPE_%", "sMAPE_%"
    ],
    "Model1_BaselineWeighted_Test": [
        mae_test_baseline_w, mse_test_baseline_w, rmse_test_baseline_w, r2_test_baseline_w,
        medae_test_baseline_w, maxerr_test_baseline_w, evs_test_baseline_w,
        acc05_test_baseline_w, acc10_test_baseline_w, mape_test_baseline_w, smape_test_baseline_w
    ],
    "Model2_RandomSearchWeighted_Test": [
        mae_test_random_w, mse_test_random_w, rmse_test_random_w, r2_test_random_w,
        medae_test_random_w, maxerr_test_random_w, evs_test_random_w,
        acc05_test_random_w, acc10_test_random_w, mape_test_random_w, smape_test_random_w
    ],
    "Model3_SmallGridWeighted_Test": [
        mae_test_grid_w, mse_test_grid_w, rmse_test_grid_w, r2_test_grid_w,
        medae_test_grid_w, maxerr_test_grid_w, evs_test_grid_w,
        acc05_test_grid_w, acc10_test_grid_w, mape_test_grid_w, smape_test_grid_w
    ],
    "Better if": [
        "Lower", "Lower", "Lower", "Higher", "Lower", "Lower", "Higher",
        "Higher", "Higher", "Lower", "Lower"
    ]
})

print("\n=== TEST Metrics Comparison (All Models) ===")
display(metrics_all_models_test)

metrics_all_models_train = pd.DataFrame({
    "Metric": [
        "MAE", "MSE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance",
        "Accuracy@±0.5 (%)", "Accuracy@±1.0 (%)"
    ],
    "Model1_BaselineWeighted_Train": [
        mae_train_baseline_w, mse_train_baseline_w, rmse_train_baseline_w, r2_train_baseline_w,
        medae_train_baseline_w, maxerr_train_baseline_w, evs_train_baseline_w,
        acc05_train_baseline_w, acc10_train_baseline_w
    ],
    "Model2_RandomSearchWeighted_Train": [
        mae_train_random_w, mse_train_random_w, rmse_train_random_w, r2_train_random_w,
        medae_train_random_w, maxerr_train_random_w, evs_train_random_w,
        acc05_train_random_w, acc10_train_random_w
    ],
    "Model3_SmallGridWeighted_Train": [
        mae_train_grid_w, mse_train_grid_w, rmse_train_grid_w, r2_train_grid_w,
        medae_train_grid_w, maxerr_train_grid_w, evs_train_grid_w,
        acc05_train_grid_w, acc10_train_grid_w
    ]
})

print("\n=== TRAIN Metrics Comparison (All Models) ===")
display(metrics_all_models_train)


# ============================================================
# 13) Choose best model for SHAP (by Test RMSE here)
#    You can change criterion to R2 or MAE if preferred
# ============================================================
model_rmse_test_map = {
    "Model1_BaselineWeighted": rmse_test_baseline_w,
    "Model2_RandomSearchWeighted": rmse_test_random_w,
    "Model3_SmallGridWeighted": rmse_test_grid_w
}

best_model_name_for_shap = min(model_rmse_test_map, key=model_rmse_test_map.get)

if best_model_name_for_shap == "Model1_BaselineWeighted":
    model_best_for_shap = model_xgb_weighted_baseline
elif best_model_name_for_shap == "Model2_RandomSearchWeighted":
    model_best_for_shap = model_xgb_weighted_randomsearch
else:
    model_best_for_shap = model_xgb_weighted_gridsearch_small

print("\nBest model selected for SHAP (by lowest Test RMSE):", best_model_name_for_shap)
print("RMSE:", model_rmse_test_map[best_model_name_for_shap])


# ============================================================
# 14) SHAP Global Feature Importance (best model)
# ============================================================
# Sample for speed (increase if needed)
shap_sample_size = min(1000, len(X_train_encoded))
X_shap_sample_best = X_train_encoded.sample(shap_sample_size, random_state=42).copy()
X_shap_sample_best.columns = X_shap_sample_best.columns.astype(str)

explainer_best = shap.TreeExplainer(model_best_for_shap)
shap_values_best = explainer_best.shap_values(X_shap_sample_best)

print("SHAP sample shape:", X_shap_sample_best.shape)
print("SHAP values shape:", np.array(shap_values_best).shape)

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values_best, X_shap_sample_best, show=False)
plt.tight_layout()
plt.show()

# Summary bar plot
plt.figure()
shap.summary_plot(shap_values_best, X_shap_sample_best, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# SHAP importance table
shap_global_importance_best_df = pd.DataFrame({
    "feature": X_shap_sample_best.columns.astype(str),
    "mean_abs_shap": np.abs(shap_values_best).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 30 Global SHAP features (best model):")
display(shap_global_importance_best_df.head(30))


# ============================================================
# 15) Predict YOUR EXAMPLE ROW with BOTH tuned models (Random + Grid)
# ============================================================
if "X_new_encoded" in globals():
    # Safety align again
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).copy()
    for c in X_new_encoded.columns:
        X_new_encoded[c] = pd.to_numeric(X_new_encoded[c], errors="coerce")
    X_new_encoded = X_new_encoded.astype(float)

    pred_new_random_w = model_xgb_weighted_randomsearch.predict(X_new_encoded)
    pred_new_grid_w = model_xgb_weighted_gridsearch_small.predict(X_new_encoded)

    pred_new_random_w_clip = np.clip(pred_new_random_w, 0, 10)
    pred_new_grid_w_clip = np.clip(pred_new_grid_w, 0, 10)

    # Optional also baseline
    pred_new_baseline_w = model_xgb_weighted_baseline.predict(X_new_encoded)
    pred_new_baseline_w_clip = np.clip(pred_new_baseline_w, 0, 10)

    predictions_example_compare_df = pd.DataFrame({
        "Model": [
            "Model1_BaselineWeighted",
            "Model2_RandomSearchWeighted",
            "Model3_SmallGridWeighted"
        ],
        "Prediction_Raw": [
            float(pred_new_baseline_w[0]),
            float(pred_new_random_w[0]),
            float(pred_new_grid_w[0])
        ],
        "Prediction_Clipped_0_10": [
            float(pred_new_baseline_w_clip[0]),
            float(pred_new_random_w_clip[0]),
            float(pred_new_grid_w_clip[0])
        ]
    })

    print("\n=== Prediction on your example row (all models) ===")
    display(predictions_example_compare_df)
else:
    print("\nX_new_encoded not found. Skipping example row prediction blocks.")


# ============================================================
# 16) SHAP Local Explanation for YOUR EXAMPLE ROW (best model)
# ============================================================
if "X_new_encoded" in globals():
    shap_values_new_best = explainer_best.shap_values(X_new_encoded)

    base_value_best = explainer_best.expected_value
    if isinstance(base_value_best, (list, np.ndarray)):
        base_value_best = float(np.array(base_value_best).reshape(-1)[0])
    else:
        base_value_best = float(base_value_best)

    row_shap_values_best = np.array(shap_values_new_best)[0]
    row_values_best = X_new_encoded.iloc[0]

    shap_local_example_best_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "feature_value": row_values_best.values,
        "shap_value": row_shap_values_best,
        "abs_shap_value": np.abs(row_shap_values_best)
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    shap_local_example_best_df["effect_on_prediction"] = np.where(
        shap_local_example_best_df["shap_value"] > 0, "pushes UP",
        np.where(shap_local_example_best_df["shap_value"] < 0, "pushes DOWN", "neutral")
    )

    print("\n=== SHAP Local Explanation for your example row (best model) ===")
    print("Best model:", best_model_name_for_shap)
    display(shap_local_example_best_df.head(30))

    # Sanity check reconstruction
    pred_example_best_raw = float(model_best_for_shap.predict(X_new_encoded)[0])
    reconstructed_pred_best = float(base_value_best + row_shap_values_best.sum())

    print("Base value (expected):", base_value_best)
    print("Sum SHAP values      :", float(row_shap_values_best.sum()))
    print("Reconstructed pred   :", reconstructed_pred_best)
    print("Model raw prediction :", pred_example_best_raw)
    print("Difference           :", abs(reconstructed_pred_best - pred_example_best_raw))

    # Waterfall plot
    shap_explanation_new_best = shap.Explanation(
        values=row_shap_values_best,
        base_values=base_value_best,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )

    plt.figure()
    shap.plots.waterfall(shap_explanation_new_best, max_display=20, show=False)
    plt.tight_layout()
    plt.show()

    # Optional: Top positive / negative contributors
    top_positive_shap_example = (
        shap_local_example_best_df.sort_values("shap_value", ascending=False)
        .head(10)[["feature", "feature_value", "shap_value"]]
    )
    top_negative_shap_example = (
        shap_local_example_best_df.sort_values("shap_value", ascending=True)
        .head(10)[["feature", "feature_value", "shap_value"]]
    )

    print("\nTop positive contributors (push prediction UP):")
    display(top_positive_shap_example)

    print("\nTop negative contributors (push prediction DOWN):")
    display(top_negative_shap_example)
else:
    print("\nX_new_encoded not found. Skipping SHAP local explanation for example row.")


# ============================================================
# 17) (Optional) SHAP Local Explanation for BOTH tuned models on your example row
#     (Random Search + Small Grid Search)
# ============================================================
if "X_new_encoded" in globals():
    # ---- Random Search tuned model local SHAP
    explainer_random_w = shap.TreeExplainer(model_xgb_weighted_randomsearch)
    shap_values_new_random_w = explainer_random_w.shap_values(X_new_encoded)
    base_value_random_w = explainer_random_w.expected_value
    if isinstance(base_value_random_w, (list, np.ndarray)):
        base_value_random_w = float(np.array(base_value_random_w).reshape(-1)[0])
    else:
        base_value_random_w = float(base_value_random_w)

    row_shap_random_w = np.array(shap_values_new_random_w)[0]
    shap_local_example_random_w_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "feature_value": X_new_encoded.iloc[0].values,
        "shap_value": row_shap_random_w,
        "abs_shap_value": np.abs(row_shap_random_w)
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    shap_local_example_random_w_df["effect_on_prediction"] = np.where(
        shap_local_example_random_w_df["shap_value"] > 0, "pushes UP",
        np.where(shap_local_example_random_w_df["shap_value"] < 0, "pushes DOWN", "neutral")
    )

    # ---- Small Grid tuned model local SHAP
    explainer_grid_w = shap.TreeExplainer(model_xgb_weighted_gridsearch_small)
    shap_values_new_grid_w = explainer_grid_w.shap_values(X_new_encoded)
    base_value_grid_w = explainer_grid_w.expected_value
    if isinstance(base_value_grid_w, (list, np.ndarray)):
        base_value_grid_w = float(np.array(base_value_grid_w).reshape(-1)[0])
    else:
        base_value_grid_w = float(base_value_grid_w)

    row_shap_grid_w = np.array(shap_values_new_grid_w)[0]
    shap_local_example_grid_w_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "feature_value": X_new_encoded.iloc[0].values,
        "shap_value": row_shap_grid_w,
        "abs_shap_value": np.abs(row_shap_grid_w)
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    shap_local_example_grid_w_df["effect_on_prediction"] = np.where(
        shap_local_example_grid_w_df["shap_value"] > 0, "pushes UP",
        np.where(shap_local_example_grid_w_df["shap_value"] < 0, "pushes DOWN", "neutral")
    )

    print("\n=== Local SHAP (Random Search tuned model) - top 20 ===")
    display(shap_local_example_random_w_df.head(20))

    print("\n=== Local SHAP (Small Grid tuned model) - top 20 ===")
    display(shap_local_example_grid_w_df.head(20))


# ============================================================
# 18) Save objects for later use
# ============================================================
# Models
saved_model_1_baseline_weighted = model_xgb_weighted_baseline
saved_model_2_randomsearch_weighted = model_xgb_weighted_randomsearch
saved_model_3_smallgrid_weighted = model_xgb_weighted_gridsearch_small

# Search objects
saved_random_search_weighted = random_search_xgb_weighted
saved_grid_search_weighted_small = grid_search_xgb_weighted_small

# Results tables
saved_metrics_all_models_test = metrics_all_models_test.copy()
saved_metrics_all_models_train = metrics_all_models_train.copy()
saved_cv_results_random_weighted = cv_results_random_weighted_df.copy()
saved_shap_global_importance_best = shap_global_importance_best_df.copy()

if "X_new_encoded" in globals():
    saved_predictions_example_compare = predictions_example_compare_df.copy()
    saved_shap_local_example_best = shap_local_example_best_df.copy()

print("\nSaved objects ✅")
print("- saved_model_1_baseline_weighted")
print("- saved_model_2_randomsearch_weighted")
print("- saved_model_3_smallgrid_weighted")
print("- saved_random_search_weighted")
print("- saved_grid_search_weighted_small")
print("- saved_metrics_all_models_test")
print("- saved_metrics_all_models_train")
print("- saved_cv_results_random_weighted")
print("- saved_shap_global_importance_best")
if "X_new_encoded" in globals():
    print("- saved_predictions_example_compare")
    print("- saved_shap_local_example_best")

```
