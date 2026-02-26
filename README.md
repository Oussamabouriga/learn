```

# ============================================================
# XGBOOST — BAYESIAN OPTIMIZATION (Optuna, MULTI-OBJECTIVE)
# CV objectives (on training folds only):
#   - minimize MAE_CV
#   - minimize RMSE_CV
#   - maximize R2_CV
#   - maximize Accuracy@±tol_CV
#
# Then:
#   - Train final model on FULL train (with sample weights)
#   - Report Train + Test metrics:
#       [mae_train, rmse_train, r2_train, acc_train]
#       [mae_test,  rmse_test,  r2_test,  acc_test]
#   - Predict example row (X_new_encoded) if available
#   - SHAP global + example
# ============================================================
# Requirements:
#   pip install optuna xgboost shap scikit-learn
# ============================================================

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
import shap
import matplotlib.pyplot as plt


# ==============================
# 1) CONFIG (editable)
# ==============================
random_state = 42

# Business range (note 0..10)
clip_min, clip_max = 0, 10

# Tolerance for regression accuracy
tol = 1.0  # change to 0.5 if you want

# Sample-weight config (imbalanced regression)
n_bins = 10
clip_min_w, clip_max_w = 0.5, 5.0

# Optuna config
n_splits = 5
n_trials = 50  # adjust 30..120 depending on budget

# How to pick ONE trial from Pareto front (editable weights)
# Score lower is better:
# score = w_mae*MAE + w_rmse*RMSE - w_r2*R2 - w_acc*(Acc/100)
score_w_mae = 1.0
score_w_rmse = 1.0
score_w_r2 = 1.0
score_w_acc = 0.5


# ==============================
# 2) ASSUMPTIONS (you already have these)
# ==============================
# X_train_encoded, X_test_encoded must be pandas DataFrames (recommended)
# y_train, y_test must be pandas Series
# X_new_encoded optional

# Safety cast
X_train_encoded = X_train_encoded.copy().astype(float)
X_test_encoded  = X_test_encoded.copy().astype(float)
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test  = pd.to_numeric(y_test, errors="coerce").astype(float)

if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)

print("Train:", X_train_encoded.shape, y_train.shape)
print("Test :", X_test_encoded.shape, y_test.shape)


# ==============================
# 3) SAMPLE WEIGHTS on y_train ONLY
# ==============================
try:
    y_bins = pd.qcut(y_train, q=min(n_bins, y_train.nunique()), duplicates="drop")
except Exception:
    y_bins = pd.cut(y_train, bins=min(n_bins, max(2, y_train.nunique())))

bin_counts = y_bins.value_counts()
weights_train = y_bins.map(lambda b: 1.0 / bin_counts[b]).astype(float).values
weights_train = weights_train / np.mean(weights_train)
weights_train = np.clip(weights_train, clip_min_w, clip_max_w)

print("\nWeights summary:")
print(pd.Series(weights_train).describe())


# ==============================
# 4) KFold CV (used by Optuna)
# ==============================
cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ==============================
# 5) MULTI-OBJECTIVE OPTUNA (CV metrics)
# ==============================
def objective(trial):
    params = {
        # fixed core params
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
        "missing": np.nan,

        # tuned params
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),

        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
    }

    fold_mae, fold_rmse, fold_r2, fold_acc = [], [], [], []

    for tr_idx, va_idx in cv.split(X_train_encoded):
        X_tr = X_train_encoded.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        w_tr = weights_train[tr_idx]

        X_va = X_train_encoded.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        pred_va = model.predict(X_va)
        pred_va = np.clip(pred_va, clip_min, clip_max)

        mae = float(mean_absolute_error(y_va, pred_va))
        rmse = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        r2 = float(r2_score(y_va, pred_va))
        acc = float((np.abs(y_va.values - pred_va) <= tol).mean() * 100)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_acc.append(acc)

    mae_cv = float(np.mean(fold_mae))
    rmse_cv = float(np.mean(fold_rmse))
    r2_cv = float(np.mean(fold_r2))
    acc_cv = float(np.mean(fold_acc))

    # Return 4 objectives for Optuna
    return mae_cv, rmse_cv, r2_cv, acc_cv


study = optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize"])
study.optimize(objective, n_trials=n_trials)

print("\n✅ Optuna multi-objective done")
print("Pareto trials:", len(study.best_trials))


# ==============================
# 6) PICK ONE trial from Pareto front (editable scoring)
# ==============================
best_trial = None
best_score = float("inf")

for t in study.best_trials:
    mae_cv, rmse_cv, r2_cv, acc_cv = t.values
    score = (
        score_w_mae * mae_cv
        + score_w_rmse * rmse_cv
        - score_w_r2 * r2_cv
        - score_w_acc * (acc_cv / 100.0)
    )
    if score < best_score:
        best_score = score
        best_trial = t

print("\n✅ Selected trial from Pareto front")
print("Score:", best_score)
print("CV metrics (MAE, RMSE, R2, Acc):", best_trial.values)
print("Best params:", best_trial.params)

best_params = best_trial.params


# ==============================
# 7) TRAIN FINAL MODEL on FULL TRAIN (weighted)
# ==============================
final_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": random_state,
    "n_jobs": -1,
    "verbosity": 0,
    "missing": np.nan,
    **best_params
}

xgb_bayes_best = XGBRegressor(**final_params)
xgb_bayes_best.fit(X_train_encoded, y_train, sample_weight=weights_train)

print("\n✅ Final Bayesian XGBoost trained")


# ==============================
# 8) REPORT TRAIN + TEST METRICS (what you asked)
# ==============================
pred_train_b = np.clip(xgb_bayes_best.predict(X_train_encoded), clip_min, clip_max)
pred_test_b  = np.clip(xgb_bayes_best.predict(X_test_encoded),  clip_min, clip_max)

mae_train_b  = float(mean_absolute_error(y_train, pred_train_b))
rmse_train_b = float(np.sqrt(mean_squared_error(y_train, pred_train_b)))
r2_train_b   = float(r2_score(y_train, pred_train_b))
acc_train_b  = float((np.abs(y_train.values - pred_train_b) <= tol).mean() * 100)

mae_test_b   = float(mean_absolute_error(y_test, pred_test_b))
rmse_test_b  = float(np.sqrt(mean_squared_error(y_test, pred_test_b)))
r2_test_b    = float(r2_score(y_test, pred_test_b))
acc_test_b   = float((np.abs(y_test.values - pred_test_b) <= tol).mean() * 100)

print("\n=== Train metrics ===")
print([mae_train_b, rmse_train_b, r2_train_b, acc_train_b])

print("\n=== Test metrics ===")
print([mae_test_b, rmse_test_b, r2_test_b, acc_test_b])

metrics_table = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train":  [mae_train_b, rmse_train_b, r2_train_b, acc_train_b],
    "Test":   [mae_test_b,  rmse_test_b,  r2_test_b,  acc_test_b],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})
display(metrics_table)


# ==============================
# 9) PREDICT EXAMPLE ROW (if available)
# ==============================
if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)
    pred_new = float(np.clip(xgb_bayes_best.predict(X_new_encoded)[0], clip_min, clip_max))
    print("\n=== Example prediction ===")
    print("Predicted note:", pred_new)
else:
    print("\nX_new_encoded not found -> skip example prediction")


# ==============================
# 10) SHAP (global + local) — clean and professional
# ==============================
# Global SHAP on sample for speed
sample_size = min(800, len(X_test_encoded))
X_shap = X_test_encoded.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(xgb_bayes_best)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP Global: summary plot")
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.show()

print("\nSHAP Global: bar plot")
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# Local SHAP for example row
if "X_new_encoded" in globals():
    shap_values_row = explainer.shap_values(X_new_encoded)
    row_shap = np.array(shap_values_row)[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    local_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "value_in_row": X_new_encoded.iloc[0].values,
        "shap_value": row_shap,
        "abs_shap": np.abs(row_shap),
        "direction": np.where(row_shap > 0, "↑ augmente", np.where(row_shap < 0, "↓ diminue", "≈ neutre"))
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("\nTop 15 drivers (example row):")
    display(local_df.head(15))

    exp = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.astype(str).tolist()
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()

    # Optional: remove E[f(X)] and f(x) labels from plot
    ax = plt.gca()
    ax.set_xlabel("")
    for t in ax.texts:
        txt = t.get_text()
        if ("E[f" in txt) or ("f(x" in txt) or ("E[f(X)]" in txt):
            t.set_visible(False)

    plt.show()
```
