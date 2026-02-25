```
# ============================================================
# XGBOOST REGRESSOR — BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna)
# + Imbalanced regression sample weights
# + K-Fold CV
# + Train final best model
# + Metrics + "accuracy@±tol"
# + Predict your example row (X_new_encoded)
# + SHAP global + SHAP local (example)
# ============================================================
# Requirements (run once if needed):
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
# 0) Assumptions (YOU ALREADY HAVE THESE)
# ==============================
# X_train_encoded, X_test_encoded: numeric matrices (float) after your encoding
# y_train, y_test: numeric target
# X_new_encoded: your example row already encoded + aligned to train columns (optional but recommended)

# Safety casting (recommended)
X_train_encoded = X_train_encoded.astype(float)
X_test_encoded  = X_test_encoded.astype(float)
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)
y_test  = pd.to_numeric(y_test, errors="coerce").astype(float)

if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)


# ==============================
# 1) Sample weights for imbalanced regression (TRAIN ONLY)
#    Inverse frequency on target bins (qcut)
# ==============================
n_bins = 10
clip_min_w, clip_max_w = 0.5, 5.0

try:
    y_bins = pd.qcut(y_train, q=min(n_bins, y_train.nunique()), duplicates="drop")
except Exception:
    y_bins = pd.cut(y_train, bins=min(n_bins, max(2, y_train.nunique())))

bin_counts = y_bins.value_counts()
weights_train = y_bins.map(lambda b: 1.0 / bin_counts[b]).astype(float).values

# Normalize + clip
weights_train = weights_train / np.mean(weights_train)
weights_train = np.clip(weights_train, clip_min_w, clip_max_w)

print("Weights summary:")
print(pd.Series(weights_train).describe())


# ==============================
# 2) KFold CV (used inside Optuna)
# ==============================
n_splits = 5
cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Business range for clipping (0..10)
clip_min, clip_max = 0, 10

# Optuna trials
n_trials = 50  # adjust 20..120 based on compute


# ==============================
# 3) Optuna objective: minimize CV RMSE (weighted)
# ==============================
def objective(trial):
    params = {
        # Core XGBoost regression
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "missing": np.nan,

        # Tuned hyperparameters (Bayesian)
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),

        # Sometimes helpful
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
    }

    fold_rmses = []

    for tr_idx, va_idx in cv.split(X_train_encoded):
        X_tr = X_train_encoded.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        w_tr = weights_train[tr_idx]

        X_va = X_train_encoded.iloc[va_idx]
        y_va = y_train.iloc[va_idx]

        model = XGBRegressor(**params)

        # weighted training
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        pred_va = model.predict(X_va)
        pred_va = np.clip(pred_va, clip_min, clip_max)

        rmse = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        fold_rmses.append(rmse)

    return float(np.mean(fold_rmses))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_params
best_cv_rmse = study.best_value

print("\n✅ Optuna done")
print("Best CV RMSE:", best_cv_rmse)
print("Best params:", best_params)


# ==============================
# 4) Train final best model on FULL train (weighted)
# ==============================
final_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "missing": np.nan,
    **best_params
}

xgb_bayes_best = XGBRegressor(**final_params)

xgb_bayes_best.fit(
    X_train_encoded,
    y_train,
    sample_weight=weights_train
)

print("\n✅ Final XGBoost Bayesian model trained")


# ==============================
# 5) Evaluate (Train/Test) + accuracy@±tol
# ==============================
pred_train = np.clip(xgb_bayes_best.predict(X_train_encoded), clip_min, clip_max)
pred_test  = np.clip(xgb_bayes_best.predict(X_test_encoded),  clip_min, clip_max)

mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
r2_train = r2_score(y_train, pred_train)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))
r2_test = r2_score(y_test, pred_test)

tol = 1.0  # change to 0.5 if you want
acc_train = (np.abs(y_train.values - pred_train) <= tol).mean() * 100
acc_test  = (np.abs(y_test.values  - pred_test)  <= tol).mean() * 100

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train":  [mae_train, rmse_train, r2_train, acc_train],
    "Test":   [mae_test,  rmse_test,  r2_test,  acc_test],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\n=== XGBoost Bayesian Best Model Metrics ===")
display(metrics_df)


# ==============================
# 6) Predict your example row (if provided)
# ==============================
if "X_new_encoded" in globals():
    X_new_encoded = X_new_encoded.reindex(columns=X_train_encoded.columns, fill_value=0).astype(float)
    pred_new = float(np.clip(xgb_bayes_best.predict(X_new_encoded)[0], clip_min, clip_max))
    print("\n=== Example prediction (Bayesian best) ===")
    print("Predicted note:", pred_new)
else:
    print("\nX_new_encoded not found -> skip example prediction")


# ==============================
# 7) SHAP — global importance + example explanation
# ==============================
# Global SHAP on a sample for speed
sample_size = min(800, len(X_test_encoded))
X_shap = X_test_encoded.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(xgb_bayes_best)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP: global feature importance (summary)")
shap.summary_plot(shap_values, X_shap, show=True)

print("\nSHAP: global feature importance (bar)")
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=True)

# SHAP local explanation for your example row
if "X_new_encoded" in globals():
    shap_values_new = explainer.shap_values(X_new_encoded)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.array(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    # Local contribution table (top 20)
    row_shap = np.array(shap_values_new)[0]
    local_df = pd.DataFrame({
        "feature": X_new_encoded.columns.astype(str),
        "feature_value": X_new_encoded.iloc[0].values,
        "shap_value": row_shap,
        "abs_shap": np.abs(row_shap),
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    print("\n=== SHAP local (example row) — top 20 ===")
    display(local_df.head(20))

    # Waterfall plot
    shap_exp = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=X_new_encoded.iloc[0].values,
        feature_names=X_new_encoded.columns.tolist()
    )
    shap.plots.waterfall(shap_exp, max_display=20)
    plt.show()
else:
    print("\nX_new_encoded not found -> skip SHAP local explanation")


```
