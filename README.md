```
# ============================================================
# ORDINAL REGRESSION (MORD) — note 0..10
# - Train / test
# - Metrics: MAE, RMSE, R2, Accuracy@±tol
# - Predict on example row (encoded)
# - Permutation importance (global)
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# MORD
import mord

# ==============================
# 1) DATA (already encoded)
# ==============================
# Uses your "no target encoding" encoded matrices
X_train_ord = X_train_encoded_no_te.copy().astype(float)
X_test_ord  = X_test_encoded_no_te.copy().astype(float)

y_train_ord = pd.to_numeric(y_train_no_te, errors="coerce").astype(int)
y_test_ord  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(int)

# Optional: ensure bounds 0..10
y_train_ord = y_train_ord.clip(0, 10)
y_test_ord  = y_test_ord.clip(0, 10)

print("Train:", X_train_ord.shape, y_train_ord.shape)
print("Test :", X_test_ord.shape,  y_test_ord.shape)
print("Unique y (train):", sorted(np.unique(y_train_ord)))

# ==============================
# 2) Train Ordinal Regression (MORD)
# ==============================
# Options:
# - LogisticAT (All-Threshold)  -> common / robust
# - LogisticIT (Immediate-Threshold)
# - LogisticSE (Squared Error like)
#
# I recommend starting with LogisticAT
ord_model = mord.LogisticAT(alpha=1.0)  # alpha = L2 regularization strength
ord_model.fit(X_train_ord, y_train_ord)

print("Ordinal model trained.")

# ==============================
# 3) Predict + clip
# ==============================
pred_train = ord_model.predict(X_train_ord)
pred_test  = ord_model.predict(X_test_ord)

# predictions are class labels already (int)
pred_train = np.clip(pred_train, 0, 10)
pred_test  = np.clip(pred_test,  0, 10)

# ==============================
# 4) Metrics + Accuracy@±tol
# ==============================
tol = 1.0  # change to 0.5 if you want

def regression_metrics(y_true, y_pred, tol=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc = float((np.abs(y_true - y_pred) <= tol).mean() * 100.0)
    return mae, rmse, r2, acc

mae_tr, rmse_tr, r2_tr, acc_tr = regression_metrics(y_train_ord, pred_train, tol=tol)
mae_te, rmse_te, r2_te, acc_te = regression_metrics(y_test_ord,  pred_test,  tol=tol)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train":  [mae_tr, rmse_tr, r2_tr, acc_tr],
    "Test":   [mae_te, rmse_te, r2_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== ORDINAL REGRESSION (MORD) METRICS ===")
display(metrics_df)

# ==============================
# 5) Predict on your example row (encoded)
# ==============================
# Priority:
# - if X_new_encoded_no_te exists: use it
# - else: you can provide a dict already encoded and we'll align columns

X_example = None

if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()
elif "example_row_encoded" in globals():
    X_example = pd.DataFrame([example_row_encoded]).copy()
else:
    print("No encoded example found. Define X_new_encoded_no_te or example_row_encoded if you want.")
    X_example = None

if X_example is not None:
    # align columns with training
    for c in X_train_ord.columns:
        if c not in X_example.columns:
            X_example[c] = 0.0
    extra_cols = [c for c in X_example.columns if c not in X_train_ord.columns]
    if len(extra_cols) > 0:
        X_example = X_example.drop(columns=extra_cols)

    X_example = X_example[X_train_ord.columns].astype(float)

    pred_ex = int(np.clip(ord_model.predict(X_example)[0], 0, 10))
    print("\n=== Example prediction (MORD) ===")
    print("Predicted note:", pred_ex)

# ==============================
# 6) Global feature importance (Permutation)
# ==============================
# MORD is linear-ish; SHAP tree isn't applicable.
# Permutation importance gives a clear ranking of influential features.
#
# We use neg MAE as scoring (higher is better), so importance is positive when MAE worsens after shuffling.
perm = permutation_importance(
    ord_model,
    X_test_ord,
    y_test_ord,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_train_ord.columns.astype(str),
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print("\nTop 20 features (Permutation importance, higher = more important):")
display(importance_df.head(20))

```
