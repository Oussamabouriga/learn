```
Perfect — this is the right next step.

For regression with imbalanced target (for example many values around one range and few rare values), we usually use sample weights so the model pays more attention to underrepresented target ranges.

Idea (simple)
	•	Common target values/ranges → smaller weight
	•	Rare target values/ranges → larger weight

This helps XGBoost not ignore rare cases.

⸻

Weighted training for imbalanced target (regression sample weights)

1) Create target bins on train only (important)

We bin y_train into ranges, then compute how frequent each bin is.

# ==============================
# 1) Create bins on y_train (train only)
# ==============================
import numpy as np
import pandas as pd

# Make sure target is numeric
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test = pd.to_numeric(y_test, errors="coerce").astype(float)

# Example for target in [0..10]
# You can adjust bins later (more/fewer bins)
bin_edges = np.array([0, 2, 4, 6, 8, 10, np.inf])

# Bin labels for train and test (for analysis only)
y_train_bins = pd.cut(y_train, bins=bin_edges, right=False, include_lowest=True)
y_test_bins  = pd.cut(y_test,  bins=bin_edges, right=False, include_lowest=True)

print("Train target bin distribution:")
display(y_train_bins.value_counts(dropna=False).sort_index())

print("\nTest target bin distribution:")
display(y_test_bins.value_counts(dropna=False).sort_index())


⸻

2) Compute sample weights from train bin frequencies

Rare bins get higher weights.

# ==============================
# 2) Compute inverse-frequency sample weights (train only)
# ==============================
train_bin_counts = y_train_bins.value_counts(dropna=False)

# Map each train row's bin -> count
train_bin_count_per_row = y_train_bins.map(train_bin_counts)

# Inverse frequency weight
# (rare bins => larger weight)
sample_weight_train = 1.0 / train_bin_count_per_row.astype(float)

# Normalize weights so average weight ~= 1 (recommended)
sample_weight_train = sample_weight_train / sample_weight_train.mean()

print("Sample weights summary:")
print(pd.Series(sample_weight_train).describe())

# Optional: inspect average weight per bin
weights_debug = pd.DataFrame({
    "y_train": y_train.values,
    "bin": y_train_bins.astype(str).values,
    "weight": np.asarray(sample_weight_train, dtype=float)
})

print("\nAverage weight per bin:")
display(weights_debug.groupby("bin")["weight"].agg(["count", "mean", "min", "max"]).sort_index())


⸻

3) Train weighted XGBoost regressor

Same model as baseline, but pass sample_weight=sample_weight_train.

# ==============================
# 3) Weighted XGBoost Regressor (imbalanced target-aware)
# ==============================
from xgboost import XGBRegressor

xgb_reg_weighted = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",

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

# Train with sample weights
xgb_reg_weighted.fit(
    X_train_encoded,
    y_train,
    sample_weight=np.asarray(sample_weight_train, dtype=float)
)

print("Weighted XGBoost model trained successfully ✅")


⸻

4) Predict (train + test)

# ==============================
# 4) Predictions (weighted model)
# ==============================
pred_train_w = xgb_reg_weighted.predict(X_train_encoded)
pred_test_w  = xgb_reg_weighted.predict(X_test_encoded)

# Optional clip to business target range (0..10)
pred_train_w_clipped = np.clip(pred_train_w, 0, 10)
pred_test_w_clipped  = np.clip(pred_test_w, 0, 10)

print("Weighted prediction done ✅")
print("Train predictions sample:", pred_train_w_clipped[:10])
print("Test predictions sample :", pred_test_w_clipped[:10])


⸻

5) Evaluate weighted model (same metrics)

# ==============================
# 5) Weighted model evaluation
# ==============================
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

# --- Train metrics
mae_train_w = mean_absolute_error(y_train, pred_train_w_clipped)
mse_train_w = mean_squared_error(y_train, pred_train_w_clipped)
rmse_train_w = np.sqrt(mse_train_w)
r2_train_w = r2_score(y_train, pred_train_w_clipped)
medae_train_w = median_absolute_error(y_train, pred_train_w_clipped)
maxerr_train_w = max_error(y_train, pred_train_w_clipped)
evs_train_w = explained_variance_score(y_train, pred_train_w_clipped)

# --- Test metrics
mae_test_w = mean_absolute_error(y_test, pred_test_w_clipped)
mse_test_w = mean_squared_error(y_test, pred_test_w_clipped)
rmse_test_w = np.sqrt(mse_test_w)
r2_test_w = r2_score(y_test, pred_test_w_clipped)
medae_test_w = median_absolute_error(y_test, pred_test_w_clipped)
maxerr_test_w = max_error(y_test, pred_test_w_clipped)
evs_test_w = explained_variance_score(y_test, pred_test_w_clipped)

eps = 1e-8
mape_test_w = np.mean(np.abs((y_test - pred_test_w_clipped) / np.maximum(np.abs(y_test), eps))) * 100
smape_test_w = np.mean(
    2.0 * np.abs(pred_test_w_clipped - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_w_clipped), eps)
) * 100

weighted_metrics = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance", "MAPE_%", "sMAPE_%"],
    "Train_weighted": [mae_train_w, mse_train_w, rmse_train_w, r2_train_w, medae_train_w, maxerr_train_w, evs_train_w, np.nan, np.nan],
    "Test_weighted":  [mae_test_w, mse_test_w, rmse_test_w, r2_test_w, medae_test_w, maxerr_test_w, evs_test_w, mape_test_w, smape_test_w]
})

print("\n=== Weighted XGBoost Regressor Results ===")
display(weighted_metrics)


⸻

6) Compare baseline vs weighted (very important)

This tells you if weighting actually helped.

# ==============================
# 6) Compare baseline vs weighted
# Requires baseline metrics variables from your previous step
# ==============================
comparison_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance", "MAPE_%", "sMAPE_%"],
    "Baseline_Test": [
        mae_test, rmse_test, r2_test, medae_test, maxerr_test, evs_test, mape_test, smape_test
    ],
    "Weighted_Test": [
        mae_test_w, rmse_test_w, r2_test_w, medae_test_w, maxerr_test_w, evs_test_w, mape_test_w, smape_test_w
    ],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher", "Lower", "Lower"]
})

print("\n=== Baseline vs Weighted (TEST) ===")
display(comparison_df)


⸻

7) Check performance by target bin (this is where weighted models often help)

Global metrics may look similar, but rare bins can improve a lot.

# ==============================
# 7) Error by target bin (TEST)
# ==============================
test_eval_bins = pd.DataFrame({
    "y_true": y_test.values,
    "y_bin": y_test_bins.astype(str).values,
    "pred_baseline": pred_test_clipped,
    "pred_weighted": pred_test_w_clipped
})

test_eval_bins["abs_err_baseline"] = np.abs(test_eval_bins["y_true"] - test_eval_bins["pred_baseline"])
test_eval_bins["abs_err_weighted"] = np.abs(test_eval_bins["y_true"] - test_eval_bins["pred_weighted"])

bin_comparison = (
    test_eval_bins
    .groupby("y_bin")
    .agg(
        count=("y_true", "size"),
        y_mean=("y_true", "mean"),
        mae_baseline=("abs_err_baseline", "mean"),
        mae_weighted=("abs_err_weighted", "mean")
    )
    .reset_index()
)

bin_comparison["improvement_weighted_vs_baseline"] = (
    bin_comparison["mae_baseline"] - bin_comparison["mae_weighted"]
)

print("\n=== Test MAE by target bin ===")
display(bin_comparison.sort_values("y_bin"))


⸻

Important notes (very useful)
	•	Create weights using train only ✅ (you did that above)
	•	Weighting may:
	•	improve rare target ranges
	•	slightly worsen common ranges
	•	change global RMSE only a little
	•	That’s why bin-level evaluation is critical

⸻

Better weighting variants (optional, try later)

If inverse frequency is too aggressive, use softer weights:

A) Square-root inverse frequency (more stable)

sample_weight_train_sqrt = 1.0 / np.sqrt(train_bin_count_per_row.astype(float))
sample_weight_train_sqrt = sample_weight_train_sqrt / sample_weight_train_sqrt.mean()

B) Cap extreme weights (prevents overfitting rare bins)

sample_weight_train_capped = np.asarray(sample_weight_train, dtype=float).copy()
sample_weight_train_capped = np.clip(sample_weight_train_capped, 0.5, 5.0)
sample_weight_train_capped = sample_weight_train_capped / sample_weight_train_capped.mean()

Then retrain with the same block using sample_weight=sample_weight_train_capped.

⸻

“Accuracy-like” score for weighted model (optional)

If you want the same tolerance metric:

tol = 1.0

acc_test_tol_baseline = (np.abs(y_test - pred_test_clipped) <= tol).mean() * 100
acc_test_tol_weighted = (np.abs(y_test - pred_test_w_clipped) <= tol).mean() * 100

print(f"Baseline tolerance accuracy (±{tol}): {acc_test_tol_baseline:.2f}%")
print(f"Weighted tolerance accuracy (±{tol}): {acc_test_tol_weighted:.2f}%")


⸻

If you want, next I can give you the Random Search block (no functions) with a small smart hyperparameter space (fast enough) for:
	•	baseline model
	•	weighted model (with sample_weight)

```
