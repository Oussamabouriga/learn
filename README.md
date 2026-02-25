```
Excellent — let’s do Step 1 now: Baseline XGBoost Regressor (normal training) using your current variables:
	•	X_train_encoded
	•	X_test_encoded
	•	y_train
	•	y_test

No functions, clean blocks, and simple explanations.

⸻

1) Import what we need for training + evaluation

# ==============================
# 1) Imports for XGBoost training + evaluation
# ==============================
from xgboost import XGBRegressor

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


⸻

2) Small check before training (important)

This helps avoid training errors.

# ==============================
# 2) Final checks before training
# ==============================
print("X_train_encoded shape:", X_train_encoded.shape)
print("X_test_encoded shape :", X_test_encoded.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

print("\nX_train dtypes (unique):")
print(X_train_encoded.dtypes.unique())

print("\nAny missing in target?")
print("y_train missing:", y_train.isna().sum())
print("y_test missing :", y_test.isna().sum())

print("\nAny non-numeric columns left in X_train?")
print(X_train_encoded.select_dtypes(exclude=[np.number]).columns.tolist())


⸻

3) What we did so far (small list — as you asked)

# ==============================
# 3) Small summary of what was done before training
# ==============================
print("""
What we did before training:
1) Cleaned data and converted selected business 0 values to np.nan
2) Split data into X (features) and y (target = evaluate_note)
3) Split train/test BEFORE encoding (to avoid leakage)
4) Applied selected encodings:
   - One-Hot Encoding (chosen columns)
   - Frequency/Count Encoding (chosen columns)
   - Target Encoding (optional, if selected)
5) Added optional log1p transforms for selected numeric columns
6) Converted all final features to numeric float (XGBoost-ready)
7) Cleaned feature names (important for XGBoost)
""")


⸻

4) Baseline XGBoost Regressor (normal training, no target weighting)

This is your first reference model.

# ==============================
# 4) Baseline XGBoost Regressor (normal training)
# ==============================
xgb_reg = XGBRegressor(
    # Regression objective
    objective="reg:squarederror",
    eval_metric="rmse",

    # Main hyperparameters (good baseline)
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

    # Missing values handling (XGBoost will handle np.nan)
    missing=np.nan,

    # Performance / reproducibility
    n_jobs=-1,
    random_state=42,
    verbosity=0
)


⸻

5) Train the model

# ==============================
# 5) Train baseline model
# ==============================
xgb_reg.fit(X_train_encoded, y_train)

print("Baseline XGBoost model trained successfully ✅")


⸻

6) Predict on train and test

# ==============================
# 6) Predictions
# ==============================
pred_train = xgb_reg.predict(X_train_encoded)
pred_test = xgb_reg.predict(X_test_encoded)

# Optional: clip predictions to your business range (0..10)
pred_train_clipped = np.clip(pred_train, 0, 10)
pred_test_clipped = np.clip(pred_test, 0, 10)

print("Prediction done ✅")
print("Train predictions sample:", pred_train_clipped[:10])
print("Test predictions sample :", pred_test_clipped[:10])


⸻

7) Evaluate with many regression metrics (baseline)

# ==============================
# 7) Evaluation metrics (baseline)
# ==============================
# --- Train metrics
mae_train = mean_absolute_error(y_train, pred_train_clipped)
mse_train = mean_squared_error(y_train, pred_train_clipped)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, pred_train_clipped)
medae_train = median_absolute_error(y_train, pred_train_clipped)
maxerr_train = max_error(y_train, pred_train_clipped)
evs_train = explained_variance_score(y_train, pred_train_clipped)

# --- Test metrics
mae_test = mean_absolute_error(y_test, pred_test_clipped)
mse_test = mean_squared_error(y_test, pred_test_clipped)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, pred_test_clipped)
medae_test = median_absolute_error(y_test, pred_test_clipped)
maxerr_test = max_error(y_test, pred_test_clipped)
evs_test = explained_variance_score(y_test, pred_test_clipped)

# Safe percentage metrics (optional)
eps = 1e-8
mape_test = np.mean(np.abs((y_test - pred_test_clipped) / np.maximum(np.abs(y_test), eps))) * 100
smape_test = np.mean(
    2.0 * np.abs(pred_test_clipped - y_test) / np.maximum(np.abs(y_test) + np.abs(pred_test_clipped), eps)
) * 100


⸻

8) Print results clearly (train vs test)

# ==============================
# 8) Print baseline results clearly
# ==============================
baseline_metrics = pd.DataFrame({
    "Metric": [
        "MAE", "MSE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance", "MAPE_%", "sMAPE_%"
    ],
    "Train": [
        mae_train, mse_train, rmse_train, r2_train, medae_train, maxerr_train, evs_train, np.nan, np.nan
    ],
    "Test": [
        mae_test, mse_test, rmse_test, r2_test, medae_test, maxerr_test, evs_test, mape_test, smape_test
    ],
    "Better if": [
        "Lower", "Lower", "Lower", "Higher", "Lower", "Lower", "Higher", "Lower", "Lower"
    ]
})

print("\n=== Baseline XGBoost Regressor Results ===")
display(baseline_metrics)


⸻

9) Quick interpretation helper (simple)

# ==============================
# 9) Quick interpretation (simple)
# ==============================
print("How to read quickly:")
print("- MAE / RMSE lower = better")
print("- R2 higher = better (1.0 best, 0 means mean baseline, <0 bad)")
print("- Compare Train vs Test:")
print("   * Train much better than Test => possible overfitting")
print("   * Train and Test close => better generalization")


⸻

10) Show top feature importance (baseline)

# ==============================
# 10) Feature importance (baseline)
# ==============================
feature_importance_df = pd.DataFrame({
    "feature": X_train_encoded.columns.astype(str),
    "importance": xgb_reg.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 important features:")
display(feature_importance_df.head(20))


⸻

11) Save baseline model + predictions (optional but useful)

# ==============================
# 11) Save baseline outputs (optional)
# ==============================
baseline_xgb_model = xgb_reg

baseline_pred_train = pred_train_clipped
baseline_pred_test = pred_test_clipped

print("Saved:")
print("- baseline_xgb_model")
print("- baseline_pred_train")
print("- baseline_pred_test")
print("- baseline_metrics")
print("- feature_importance_df")


⸻

What’s next (recommended)

Next step should be exactly what you asked before:

2) Baseline XGBoost with target imbalance handling (sample weights for regression)
So you compare:
	•	Normal baseline
	•	Weighted baseline (imbalanced target-aware)

Then after that:
	•	Random Search (small, efficient)
	•	Small Grid Search (focused hyperparameters only)

If you want, I can write the weighted training block next (no functions) in the same style.

```
