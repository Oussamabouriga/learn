```

# ============================================================
# ORDINAL REGRESSION (MORD) — FIX NaN
# ============================================================

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

import mord


# ==============================
# 1) DATA (already encoded)
# ==============================
X_train_ord = X_train_encoded_no_te.copy().astype(float)
X_test_ord  = X_test_encoded_no_te.copy().astype(float)

y_train_ord = pd.to_numeric(y_train_no_te, errors="coerce").astype(int).clip(0, 10)
y_test_ord  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(int).clip(0, 10)

# Safety: convert inf -> NaN
X_train_ord = X_train_ord.replace([np.inf, -np.inf], np.nan)
X_test_ord  = X_test_ord.replace([np.inf, -np.inf], np.nan)

print("NaN count train:", int(np.isnan(X_train_ord.to_numpy()).sum()))
print("NaN count test :", int(np.isnan(X_test_ord.to_numpy()).sum()))

# ==============================
# 2) Imputation (CRITICAL for MORD)
# Fit on train only, apply to train/test
# ==============================
imputer = SimpleImputer(strategy="median")  # good default for numeric encoded data

X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train_ord),
    columns=X_train_ord.columns,
    index=X_train_ord.index
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test_ord),
    columns=X_test_ord.columns,
    index=X_test_ord.index
)

print("After impute NaN train:", int(np.isnan(X_train_imp.to_numpy()).sum()))
print("After impute NaN test :", int(np.isnan(X_test_imp.to_numpy()).sum()))

# ==============================
# 3) Train Ordinal Regression (MORD)
# ==============================
ord_model = mord.LogisticAT(alpha=1.0)
ord_model.fit(X_train_imp, y_train_ord)

print("Ordinal model trained.")

# ==============================
# 4) Predict + clip
# ==============================
pred_train = np.clip(ord_model.predict(X_train_imp), 0, 10)
pred_test  = np.clip(ord_model.predict(X_test_imp),  0, 10)

# ==============================
# 5) Metrics + Accuracy@±tol
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
# 6) Predict on your example row (encoded)
# ==============================
X_example = None

if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()
elif "example_row_encoded" in globals():
    X_example = pd.DataFrame([example_row_encoded]).copy()

if X_example is not None:
    # align columns with training
    for c in X_train_ord.columns:
        if c not in X_example.columns:
            X_example[c] = 0.0
    extra_cols = [c for c in X_example.columns if c not in X_train_ord.columns]
    if len(extra_cols) > 0:
        X_example = X_example.drop(columns=extra_cols)

    X_example = X_example[X_train_ord.columns].astype(float).replace([np.inf, -np.inf], np.nan)

    # apply SAME imputer
    X_example_imp = pd.DataFrame(
        imputer.transform(X_example),
        columns=X_example.columns,
        index=X_example.index
    )

    pred_ex = int(np.clip(ord_model.predict(X_example_imp)[0], 0, 10))
    print("\n=== Example prediction (MORD) ===")
    print("Predicted note:", pred_ex)

# ==============================
# 7) Global feature importance (Permutation)
# ==============================
perm = permutation_importance(
    ord_model,
    X_test_imp,
    y_test_ord,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_train_imp.columns.astype(str),
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print("\nTop 20 features (Permutation importance, higher = more important):")
display(importance_df.head(20))
```
