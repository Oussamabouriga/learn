```
# ============================================================
# ORDINAL REGRESSION (MORD) + BAYESIAN OPTIMIZATION (Optuna)
# - works with encoded data (NO TE)
# - imputation is mandatory (MORD doesn't accept NaN)
# - CV objective: minimize MAE/RMSE, maximize R2/Accuracy@±tol (aggregated score)
# ============================================================

import numpy as np
import pandas as pd

import mord
import optuna

from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# ==============================
# 0) CONFIG (editable)
# ==============================
random_state = 42
n_splits = 5
n_trials = 40      # increase if you want better search
alpha_min = 1e-3
alpha_max = 1e2

# We will treat target as ordinal classes 0..10
y_min, y_max = 0, 10

# weights inside the aggregated objective (tune if you want)
W_MAE = 1.0
W_RMSE = 0.7
W_R2 = 0.4
W_ACC = 0.4


# ==============================
# 1) DATA (already encoded)
# ==============================
X_train_ord = X_train_encoded_no_te.copy().astype(float)
X_test_ord  = X_test_encoded_no_te.copy().astype(float)

y_train_ord = pd.to_numeric(y_train_no_te, errors="coerce").astype(int).clip(y_min, y_max)
y_test_ord  = pd.to_numeric(y_test_no_te,  errors="coerce").astype(int).clip(y_min, y_max)

# safety: inf -> NaN
X_train_ord = X_train_ord.replace([np.inf, -np.inf], np.nan)
X_test_ord  = X_test_ord.replace([np.inf, -np.inf], np.nan)

print("Train:", X_train_ord.shape, y_train_ord.shape)
print("Test :", X_test_ord.shape,  y_test_ord.shape)


# ==============================
# 2) Metrics helper
# ==============================
def regression_metrics(y_true, y_pred, tol=1.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc = float((np.abs(y_true - y_pred) <= tol).mean() * 100.0)
    return mae, rmse, r2, acc


def aggregated_score(mae, rmse, r2, acc):
    """
    Lower is better.
    - minimize mae, rmse
    - maximize r2, acc  -> subtract them
    """
    return (W_MAE * mae) + (W_RMSE * rmse) - (W_R2 * r2) - (W_ACC * (acc / 100.0))


# ==============================
# 3) Optuna objective (CV)
# ==============================
cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def objective(trial):
    # alpha is main knob in MORD
    alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)

    # optional: tune tolerance used in Accuracy@±tol
    tol = trial.suggest_float("tol", 0.5, 1.5)

    fold_scores = []

    for tr_idx, va_idx in cv.split(X_train_ord):
        X_tr = X_train_ord.iloc[tr_idx]
        y_tr = y_train_ord.iloc[tr_idx]
        X_va = X_train_ord.iloc[va_idx]
        y_va = y_train_ord.iloc[va_idx]

        # IMPORTANT: fit imputer only on fold train
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)

        model = mord.LogisticAT(alpha=alpha)
        model.fit(X_tr_imp, y_tr)

        pred_va = np.clip(model.predict(X_va_imp), y_min, y_max)

        mae, rmse, r2, acc = regression_metrics(y_va, pred_va, tol=tol)
        score = aggregated_score(mae, rmse, r2, acc)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_params
best_score = study.best_value

print("\n=== OPTUNA DONE (MORD) ===")
print("Best aggregated score:", best_score)
print("Best params:", best_params)


# ==============================
# 4) Train FINAL model on full train with best params
# ==============================
best_alpha = float(best_params["alpha"])
best_tol = float(best_params["tol"])

imputer = SimpleImputer(strategy="median")
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

ord_model_optuna = mord.LogisticAT(alpha=best_alpha)
ord_model_optuna.fit(X_train_imp, y_train_ord)

print("\nFinal MORD trained with alpha =", best_alpha, "| tol =", best_tol)


# ==============================
# 5) Evaluate Train/Test
# ==============================
pred_train = np.clip(ord_model_optuna.predict(X_train_imp), y_min, y_max)
pred_test  = np.clip(ord_model_optuna.predict(X_test_imp),  y_min, y_max)

mae_tr, rmse_tr, r2_tr, acc_tr = regression_metrics(y_train_ord, pred_train, tol=best_tol)
mae_te, rmse_te, r2_te, acc_te = regression_metrics(y_test_ord,  pred_test,  tol=best_tol)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{best_tol:.2f}"],
    "Train":  [mae_tr, rmse_tr, r2_tr, acc_tr],
    "Test":   [mae_te, rmse_te, r2_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== MORD (Optuna) METRICS ===")
display(metrics_df)


# ==============================
# 6) Example prediction (if available)
# ==============================
if "X_new_encoded_no_te" in globals():
    X_example = X_new_encoded_no_te.copy()

    # align columns
    for c in X_train_ord.columns:
        if c not in X_example.columns:
            X_example[c] = 0.0
    extra_cols = [c for c in X_example.columns if c not in X_train_ord.columns]
    if len(extra_cols) > 0:
        X_example = X_example.drop(columns=extra_cols)

    X_example = X_example[X_train_ord.columns].astype(float).replace([np.inf, -np.inf], np.nan)
    X_example_imp = pd.DataFrame(
        imputer.transform(X_example),
        columns=X_example.columns,
        index=X_example.index
    )

    pred_ex = int(np.clip(ord_model_optuna.predict(X_example_imp)[0], y_min, y_max))
    print("\n=== Example prediction (MORD Optuna) ===")
    print("Predicted note:", pred_ex)


# ==============================
# 7) Global importance (Permutation) — works for any model
# ==============================
perm = permutation_importance(
    ord_model_optuna,
    X_test_imp,
    y_test_ord,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=random_state,
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
