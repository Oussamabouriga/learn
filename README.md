```
# ==============================
# 1) Imports
# ==============================
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

# ==============================
# 2) Make sure all X are numeric
# ==============================
X_train_encoded = X_train_encoded.copy()
X_test_encoded  = X_test_encoded.copy()

for c in X_train_encoded.columns:
    X_train_encoded[c] = pd.to_numeric(X_train_encoded[c], errors="coerce")
for c in X_test_encoded.columns:
    X_test_encoded[c] = pd.to_numeric(X_test_encoded[c], errors="coerce")

# (optional but recommended) cast to float
X_train_encoded = X_train_encoded.astype(float)
X_test_encoded  = X_test_encoded.astype(float)

# Target numeric
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test  = pd.to_numeric(y_test, errors="coerce").astype(float)

# Drop rows with missing target if any
train_mask = y_train.notna()
test_mask  = y_test.notna()

X_train_encoded = X_train_encoded.loc[train_mask].copy()
y_train = y_train.loc[train_mask].copy()

X_test_encoded = X_test_encoded.loc[test_mask].copy()
y_test = y_test.loc[test_mask].copy()

# ==============================
# 3) Baseline XGBoost Regressor (normal training, no target balancing)
# ==============================
xgb_reg = XGBRegressor(
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

    n_jobs=-1,
    random_state=42,
    verbosity=0
)

# Train
xgb_reg.fit(X_train_encoded, y_train)

# Predict
pred_train = xgb_reg.predict(X_train_encoded)
pred_test  = xgb_reg.predict(X_test_encoded)

# Optional clip to [0,10] if your score is 0..10
pred_train = np.clip(pred_train, 0, 10)
pred_test  = np.clip(pred_test, 0, 10)

# ==============================
# 4) Evaluate
# ==============================
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test, pred_test))

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE", "RMSE", "MSE", "R2", "MedianAE", "MaxError", "ExplainedVariance"
    ],
    "Train": [
        mean_absolute_error(y_train, pred_train),
        rmse_train,
        mean_squared_error(y_train, pred_train),
        r2_score(y_train, pred_train),
        median_absolute_error(y_train, pred_train),
        max_error(y_train, pred_train),
        explained_variance_score(y_train, pred_train),
    ],
    "Test": [
        mean_absolute_error(y_test, pred_test),
        rmse_test,
        mean_squared_error(y_test, pred_test),
        r2_score(y_test, pred_test),
        median_absolute_error(y_test, pred_test),
        max_error(y_test, pred_test),
        explained_variance_score(y_test, pred_test),
    ],
    "Better": [
        "Lower", "Lower", "Lower", "Higher", "Lower", "Lower", "Higher"
    ]
})

print("=== XGBoost Regressor (Normal Training) ===")
print(metrics_df.to_string(index=False))

```
