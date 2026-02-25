```

Nice — this is the right next step.

Random Search first is exactly the best choice here (much cheaper than full Grid Search), especially with XGBoost + many hyperparameters.

Below is a full no-functions workflow that includes:
	•	✅ RandomizedSearchCV for XGBoost Regressor
	•	✅ works with your weighted training (imbalanced regression via sample_weight)
	•	✅ evaluation metrics (“accuracy” equivalent for regression included)
	•	✅ SHAP (global + local)
	•	✅ prediction + SHAP explanation for your exact example row

⸻

1) Imports for Random Search + evaluation + SHAP

# ==============================
# 1) Imports (Random Search + metrics + SHAP)
# ==============================
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


⸻

2) Final checks before Random Search

# ==============================
# 2) Final checks
# ==============================
print("X_train_encoded:", X_train_encoded.shape)
print("X_test_encoded :", X_test_encoded.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nNon-numeric columns in X_train_encoded:")
print(X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist())

# Safety: all numeric float
for c in X_train_encoded.columns:
    X_train_encoded[c] = pd.to_numeric(X_train_encoded[c], errors="coerce")
for c in X_test_encoded.columns:
    X_test_encoded[c] = pd.to_numeric(X_test_encoded[c], errors="coerce")

X_train_encoded = X_train_encoded.astype(float)
X_test_encoded = X_test_encoded.astype(float)

y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test = pd.to_numeric(y_test, errors="coerce").astype(float)

print("\nDone final numeric cast ✅")


⸻

3) Regression sample weights for imbalanced target (train only)

You asked for imbalanced target handling in regression.
This is a clean practical approach: give more weight to rare target regions.

Version used here:
	•	bin y_train into quantiles
	•	compute inverse-frequency weights
	•	normalize weights (mean ≈ 1)

# ==============================
# 3) Sample weights for imbalanced regression target (TRAIN ONLY)
# ==============================
# Number of bins (adjust 5-20 depending on size)
n_bins = 10

# qcut can fail if too many duplicate target values -> duplicates='drop'
y_train_bins = pd.qcut(y_train, q=n_bins, duplicates="drop")

# Frequency per bin
bin_counts = y_train_bins.value_counts(dropna=False)

# Inverse frequency weight per bin
bin_weight_map = (1.0 / bin_counts).to_dict()

# Map each train row to a weight
sample_weight_train = y_train_bins.map(bin_weight_map).astype(float)

# Normalize weights so mean weight ~ 1 (more stable training)
sample_weight_train = sample_weight_train / sample_weight_train.mean()

print("Sample weights created ✅")
print("Min weight:", float(sample_weight_train.min()))
print("Max weight:", float(sample_weight_train.max()))
print("Mean weight:", float(sample_weight_train.mean()))

display(pd.DataFrame({
    "y_train": y_train.values[:10],
    "weight": sample_weight_train.values[:10]
}))


⸻

4) Build base XGBoost model for Random Search

# ==============================
# 4) Base XGBRegressor for Random Search
# ==============================
xgb_rs_base = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",     # fast
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    missing=np.nan
)


⸻

5) Random Search parameter space (focused, not too huge)

This is the important part.
We keep a focused search space so compute stays reasonable.

# ==============================
# 5) Random Search parameter distributions (focused)
# ==============================
param_distributions = {
    # Core complexity / learning
    "n_estimators": [200, 300, 500, 700, 900],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 2, 3, 5, 7, 10],

    # Split control
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],

    # Sampling (helps generalization)
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.8, 1.0],
    "colsample_bynode": [0.8, 1.0],

    # Regularization
    "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],

    # Tree growth mode
    "grow_policy": ["depthwise", "lossguide"],
    "max_leaves": [0, 31, 63, 127]  # used mainly when lossguide
}


⸻

6) Cross-validation setup + RandomizedSearchCV (weighted)

Why neg_root_mean_squared_error?

Good default for regression, same spirit as your eval metric RMSE.

# ==============================
# 6) Randomized Search CV (weighted training)
# ==============================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xgb_weighted = RandomizedSearchCV(
    estimator=xgb_rs_base,
    param_distributions=param_distributions,
    n_iter=30,                    # start 20-40; increase later if needed
    scoring="neg_root_mean_squared_error",
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# IMPORTANT: pass sample_weight into fit()
random_search_xgb_weighted.fit(
    X_train_encoded,
    y_train,
    sample_weight=sample_weight_train.values
)

print("Random Search completed ✅")
print("Best CV score (neg RMSE):", random_search_xgb_weighted.best_score_)
print("Best CV RMSE:", -random_search_xgb_weighted.best_score_)
print("\nBest params:")
print(random_search_xgb_weighted.best_params_)


⸻

7) Best model from Random Search + train/test predictions

# ==============================
# 7) Get best model + predictions
# ==============================
best_xgb_random_weighted = random_search_xgb_weighted.best_estimator_

pred_train_rs_w = best_xgb_random_weighted.predict(X_train_encoded)
pred_test_rs_w = best_xgb_random_weighted.predict(X_test_encoded)

# Clip to business range (0..10) if your target is a note / score
pred_train_rs_w_clipped = np.clip(pred_train_rs_w, 0, 10)
pred_test_rs_w_clipped = np.clip(pred_test_rs_w, 0, 10)

print("Predictions done ✅")
print("Train sample predictions:", pred_train_rs_w_clipped[:10])
print("Test sample predictions :", pred_test_rs_w_clipped[:10])


⸻

8) “Accuracy” for regression + full evaluation metrics

In regression, there is no classic classification accuracy.
So we add practical regression accuracy-style metrics:
	•	R² (main “accuracy-like” metric for regression)
	•	Accuracy within tolerance (e.g. ±0.5, ±1.0 note points)
	•	plus MAE/RMSE/etc.

# ==============================
# 8) Evaluation metrics + regression "accuracy"
# ==============================

# --- Train metrics
mae_train_rs_w = mean_absolute_error(y_train, pred_train_rs_w_clipped)
mse_train_rs_w = mean_squared_error(y_train, pred_train_rs_w_clipped)
rmse_train_rs_w = np.sqrt(mse_train_rs_w)
r2_train_rs_w = r2_score(y_train, pred_train_rs_w_clipped)
medae_train_rs_w = median_absolute_error(y_train, pred_train_rs_w_clipped)
maxerr_train_rs_w = max_error(y_train, pred_train_rs_w_clipped)
evs_train_rs_w = explained_variance_score(y_train, pred_train_rs_w_clipped)

# --- Test metrics
mae_test_rs_w = mean_absolute_error(y_test, pred_test_rs_w_clipped)
mse_test_rs_w = mean_squared_error(y_test, pred_test_rs_w_clipped)
rmse_test_rs_w = np.sqrt(mse_test_rs_w)
r2_test_rs_w = r2_score(y_test, pred_test_rs_w_clipped)
medae_test_rs_w = median_absolute_error(y_test, pred_test_rs_w_clipped)
maxerr_test_rs_w = max_error(y_test, pred_test_rs_w_clipped)
evs_test_rs_w = explained_variance_score(y_test, pred_test_rs_w_clipped)

# --- Regression "accuracy" by tolerance (very useful for note prediction 0..10)
tol_05_train = np.mean(np.abs(y_train - pred_train_rs_w_clipped) <= 0.5) * 100
tol_10_train = np.mean(np.abs(y_train - pred_train_rs_w_clipped) <= 1.0) * 100

tol_05_test = np.mean(np.abs(y_test - pred_test_rs_w_clipped) <= 0.5) * 100
tol_10_test = np.mean(np.abs(y_test - pred_test_rs_w_clipped) <= 1.0) * 100

# Optional percentage metrics
eps = 1e-8
mape_test_rs_w = np.mean(np.abs((y_test - pred_test_rs_w_clipped) / np.maximum(np.abs(y_test), eps))) * 100
smape_test_rs_w = np.mean(
    2.0 * np.abs(pred_test_rs_w_clipped - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_rs_w_clipped), eps)
) * 100

Print metrics clearly

# ==============================
# 9) Print metrics table (Random Search weighted)
# ==============================
random_search_weighted_metrics = pd.DataFrame({
    "Metric": [
        "MAE", "MSE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance",
        "Accuracy@±0.5 (%)", "Accuracy@±1.0 (%)", "MAPE_%", "sMAPE_%"
    ],
    "Train": [
        mae_train_rs_w, mse_train_rs_w, rmse_train_rs_w, r2_train_rs_w, medae_train_rs_w, maxerr_train_rs_w, evs_train_rs_w,
        tol_05_train, tol_10_train, np.nan, np.nan
    ],
    "Test": [
        mae_test_rs_w, mse_test_rs_w, rmse_test_rs_w, r2_test_rs_w, medae_test_rs_w, maxerr_test_rs_w, evs_test_rs_w,
        tol_05_test, tol_10_test, mape_test_rs_w, smape_test_rs_w
    ],
    "Better if": [
        "Lower", "Lower", "Lower", "Higher", "Lower", "Lower", "Higher",
        "Higher", "Higher", "Lower", "Lower"
    ]
})

print("\n=== Random Search Weighted XGBoost Results ===")
display(random_search_weighted_metrics)


⸻

9.1) Which hyperparameters mattered most in Random Search (from CV results)

This is not perfect causal importance, but very useful.

# ==============================
# 9.1) Random Search results inspection (top trials)
# ==============================
cv_results_df = pd.DataFrame(random_search_xgb_weighted.cv_results_).sort_values(
    "rank_test_score"
).reset_index(drop=True)

cols_to_show = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "param_n_estimators",
    "param_learning_rate",
    "param_max_depth",
    "param_min_child_weight",
    "param_gamma",
    "param_subsample",
    "param_colsample_bytree",
    "param_reg_alpha",
    "param_reg_lambda",
    "param_grow_policy",
    "param_max_leaves"
]

print("Top 10 Random Search trials:")
display(cv_results_df[cols_to_show].head(10))


⸻

10) SHAP global feature importance (best random-search weighted model)

# ==============================
# 10) SHAP - Global feature importance (best random-search weighted model)
# ==============================
# Sample for speed
shap_sample_size = min(1000, len(X_train_encoded))
X_shap_sample_rs_w = X_train_encoded.sample(shap_sample_size, random_state=42).copy()

# Ensure safe feature names
X_shap_sample_rs_w.columns = X_shap_sample_rs_w.columns.astype(str)

# Explainer
explainer_rs_w = shap.TreeExplainer(best_xgb_random_weighted)

# SHAP values
shap_values_rs_w = explainer_rs_w.shap_values(X_shap_sample_rs_w)

print("SHAP sample shape:", X_shap_sample_rs_w.shape)
print("SHAP values shape:", np.array(shap_values_rs_w).shape)

SHAP summary plots

# ==============================
# 11) SHAP summary plots (global)
# ==============================
plt.figure()
shap.summary_plot(shap_values_rs_w, X_shap_sample_rs_w, show=False)
plt.tight_layout()
plt.show()

plt.figure()
shap.summary_plot(shap_values_rs_w, X_shap_sample_rs_w, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

SHAP importance dataframe

# ==============================
# 12) Global SHAP importance dataframe
# ==============================
mean_abs_shap_rs_w = np.abs(shap_values_rs_w).mean(axis=0)

shap_importance_random_weighted_df = pd.DataFrame({
    "feature": X_shap_sample_rs_w.columns.astype(str),
    "mean_abs_shap": mean_abs_shap_rs_w
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("Top 20 global SHAP features (best random-search weighted model):")
display(shap_importance_random_weighted_df.head(20))


⸻

11) Predict your exact example row (same row you gave before)

You asked to include your example again.
This block assumes you already created and encoded it before into:
	•	X_new_encoded

If not, use the encoding block we built earlier first, then run this.

# ==============================
# 13) Predict your example row using best random-search weighted model
# ==============================
print("X_new_encoded.shape:", X_new_encoded.shape)
print("Expected columns:", X_train_encoded.shape[1])

# Safety: align columns exactly (in case anything changed)
X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).copy()

for c in X_new_encoded.columns:
    X_new_encoded[c] = pd.to_numeric(X_new_encoded[c], errors="coerce")
X_new_encoded = X_new_encoded.astype(float)

pred_new_rs_w = best_xgb_random_weighted.predict(X_new_encoded)
pred_new_rs_w_clipped = np.clip(pred_new_rs_w, 0, 10)

print("Random Search Weighted prediction (raw):", float(pred_new_rs_w[0]))
print("Random Search Weighted prediction (clipped 0..10):", float(pred_new_rs_w_clipped[0]))


⸻

12) SHAP explanation for your example row (why this prediction)

# ==============================
# 14) SHAP for your example row (local explanation)
# ==============================
# SHAP values for the single row
shap_values_new_rs_w = explainer_rs_w.shap_values(X_new_encoded)

# Expected/base value
base_value_rs_w = explainer_rs_w.expected_value
if isinstance(base_value_rs_w, (list, np.ndarray)):
    base_value_rs_w = np.array(base_value_rs_w).reshape(-1)[0]

row_shap_values_rs_w = np.array(shap_values_new_rs_w)[0]
row_values_rs_w = X_new_encoded.iloc[0]

# Contribution dataframe
shap_row_random_weighted_df = pd.DataFrame({
    "feature": X_new_encoded.columns.astype(str),
    "feature_value": row_values_rs_w.values,
    "shap_value": row_shap_values_rs_w,
    "abs_shap_value": np.abs(row_shap_values_rs_w)
}).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

shap_row_random_weighted_df["effect_on_prediction"] = np.where(
    shap_row_random_weighted_df["shap_value"] > 0, "pushes UP",
    np.where(shap_row_random_weighted_df["shap_value"] < 0, "pushes DOWN", "neutral")
)

print("Top features that influenced THIS example prediction:")
display(shap_row_random_weighted_df.head(25))

# Sanity check
reconstructed_pred_rs_w = float(base_value_rs_w + row_shap_values_rs_w.sum())

print("Base value (expected):", float(base_value_rs_w))
print("Sum SHAP values      :", float(row_shap_values_rs_w.sum()))
print("Reconstructed pred   :", reconstructed_pred_rs_w)
print("Model raw prediction :", float(pred_new_rs_w[0]))
print("Difference           :", abs(reconstructed_pred_rs_w - float(pred_new_rs_w[0])))

Waterfall plot for your example row

# ==============================
# 15) SHAP waterfall plot for your example row
# ==============================
shap_exp_new_rs_w = shap.Explanation(
    values=row_shap_values_rs_w,
    base_values=base_value_rs_w,
    data=X_new_encoded.iloc[0].values,
    feature_names=X_new_encoded.columns.tolist()
)

plt.figure()
shap.plots.waterfall(shap_exp_new_rs_w, max_display=20, show=False)
plt.tight_layout()
plt.show()


⸻

13) Optional: compare baseline / weighted / random-search-weighted on the same example

# ==============================
# 16) Compare models on the same example row (optional)
# ==============================
comparison_pred = []

if "baseline_xgb_model" in globals():
    p = baseline_xgb_model.predict(X_new_encoded)
    comparison_pred.append({
        "model": "Baseline XGB",
        "prediction_raw": float(p[0]),
        "prediction_clipped_0_10": float(np.clip(p[0], 0, 10))
    })

if "xgb_reg_weighted" in globals():
    p = xgb_reg_weighted.predict(X_new_encoded)
    comparison_pred.append({
        "model": "Weighted XGB (manual params)",
        "prediction_raw": float(p[0]),
        "prediction_clipped_0_10": float(np.clip(p[0], 0, 10))
    })

p = best_xgb_random_weighted.predict(X_new_encoded)
comparison_pred.append({
    "model": "Best RandomSearch Weighted XGB",
    "prediction_raw": float(p[0]),
    "prediction_clipped_0_10": float(np.clip(p[0], 0, 10))
})

comparison_pred_df = pd.DataFrame(comparison_pred)
display(comparison_pred_df)


⸻

14) Save outputs (useful for next step)

# ==============================
# 17) Save important outputs
# ==============================
best_random_search_weighted_model = best_xgb_random_weighted
best_random_search_weighted_params = random_search_xgb_weighted.best_params_
best_random_search_weighted_cv_results = cv_results_df.copy()

pred_test_best_random_weighted = pred_test_rs_w_clipped.copy()
metrics_best_random_weighted = random_search_weighted_metrics.copy()
shap_global_best_random_weighted = shap_importance_random_weighted_df.copy()
shap_local_example_best_random_weighted = shap_row_random_weighted_df.copy()

print("Saved ✅")
print("- best_random_search_weighted_model")
print("- best_random_search_weighted_params")
print("- best_random_search_weighted_cv_results")
print("- pred_test_best_random_weighted")
print("- metrics_best_random_weighted")
print("- shap_global_best_random_weighted")
print("- shap_local_example_best_random_weighted")


⸻

Quick notes (important)
	•	For regression, “accuracy” is not the same as classification.
Use:
	•	R²
	•	Accuracy@±0.5
	•	Accuracy@±1.0
	•	Random Search is much better than Grid Search at this stage because:
	•	too many hyperparameters
	•	XGBoost training is expensive
	•	you get strong results with less compute

⸻

If you want next (recommended)

I can write the next block for:

Small Focused Grid Search (after Random Search)

Only around the best params, e.g. tune just:
	•	max_depth
	•	min_child_weight
	•	learning_rate
	•	subsample
	•	colsample_bytree

That gives you a cheap refinement step after Random Search.
```
