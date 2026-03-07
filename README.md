```

# ============================================================
# XGBoost Regressor — WEIGHTED + BAYESIAN OPTIMIZATION (Optuna)
# Multi-objective in CV:
#   minimize: MAE_CV, RMSE_CV
#   maximize: R2_CV, Accuracy@±tol_CV
#
# Then:
# - select ONE best trial from Pareto front (editable scoring)
# - train final model on full train with sample_weight
# - evaluate on train/test (MAE, RMSE, R2, Accuracy@±tol)
# - SHAP global + SHAP local (example)
# - save model to: models/xgboost/regression/<model_name>/
#
# Inputs expected (already prepared):
#   X_train_xgboost_reg_w, X_test_xgboost_reg_w
#   y_train_xgboost_reg_w, y_test_xgboost_reg_w
#   sample_weight_train_xgboost_reg_w
# Example row (already prepared/encoded):
#   X_new_xgboost_reg_w OR X_new_encoded_no_te
#
# Requirements:
#   pip install optuna shap xgboost scikit-learn
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Config (editable)
# ==============================
random_state = 42
n_splits = 5
n_trials = 60           # increase if you have compute
tol = 1.0               # Accuracy@±tol

clip_pred_to_0_10 = True

# Pareto selection scoring (editable)
# We want: low MAE, low RMSE, high R2, high Accuracy@tol
w_mae = 1.0
w_rmse = 1.0
w_r2 = 1.0
w_acc = 1.0


# ==============================
# 2) Ensure correct dtypes
# ==============================
X_train_bo = X_train_xgboost_reg_w.copy().astype(float)
X_test_bo  = X_test_xgboost_reg_w.copy().astype(float)

y_train_bo = pd.to_numeric(y_train_xgboost_reg_w, errors="coerce").astype(float)
y_test_bo  = pd.to_numeric(y_test_xgboost_reg_w,  errors="coerce").astype(float)

w_train_bo = np.asarray(sample_weight_train_xgboost_reg_w, dtype=float)

# Drop missing target rows (if any)
mask_tr = y_train_bo.notna()
X_train_bo = X_train_bo.loc[mask_tr].copy()
y_train_bo = y_train_bo.loc[mask_tr].copy()
w_train_bo = w_train_bo[mask_tr.values]

mask_te = y_test_bo.notna()
X_test_bo = X_test_bo.loc[mask_te].copy()
y_test_bo = y_test_bo.loc[mask_te].copy()

cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ==============================
# 3) Multi-objective objective (CV)
# ==============================
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": 0,
        "missing": np.nan,

        # Tuned params (Bayesian)
        "n_estimators": trial.suggest_int("n_estimators", 300, 1400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }

    maes, rmses, r2s, accs = [], [], [], []

    for tr_idx, va_idx in cv.split(X_train_bo):
        X_tr = X_train_bo.iloc[tr_idx]
        y_tr = y_train_bo.iloc[tr_idx]
        w_tr = w_train_bo[tr_idx]

        X_va = X_train_bo.iloc[va_idx]
        y_va = y_train_bo.iloc[va_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        pred_va = model.predict(X_va)
        if clip_pred_to_0_10:
            pred_va = np.clip(pred_va, 0, 10)

        mae = float(mean_absolute_error(y_va, pred_va))
        rmse = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        r2 = float(r2_score(y_va, pred_va))
        acc = float((np.abs(y_va.values - pred_va) <= tol).mean() * 100.0)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        accs.append(acc)

    mae_cv = float(np.mean(maes))
    rmse_cv = float(np.mean(rmses))
    r2_cv = float(np.mean(r2s))
    acc_cv = float(np.mean(accs))

    # Multi-objective: minimize MAE/RMSE, maximize R2/ACC
    return mae_cv, rmse_cv, r2_cv, acc_cv


study = optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize"])
study.optimize(objective, n_trials=n_trials)

print("\nBayesian optimization finished")
print("Pareto trials:", len(study.best_trials))


# ==============================
# 4) Pick ONE best trial from Pareto front (editable scoring)
# ==============================
best_trial = None
best_score = -1e18

for t in study.best_trials:
    mae_cv, rmse_cv, r2_cv, acc_cv = t.values

    # Higher score is better: we subtract errors and add good metrics
    score = (-w_mae * mae_cv) + (-w_rmse * rmse_cv) + (w_r2 * r2_cv) + (w_acc * acc_cv)

    if score > best_score:
        best_score = score
        best_trial = t

print("\nSelected trial from Pareto front")
print("Score:", best_score)
print("CV values (MAE, RMSE, R2, Acc@tol):", best_trial.values)
print("Best params:", best_trial.params)

best_params = best_trial.params


# ==============================
# 5) Train final model on full train (weighted)
# ==============================
final_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": random_state,
    "verbosity": 0,
    "missing": np.nan,
    **best_params
}

xgboost_reg_weighted_bayes = XGBRegressor(**final_params)
xgboost_reg_weighted_bayes.fit(X_train_bo, y_train_bo, sample_weight=w_train_bo)

print("\nFinal model trained")


# ==============================
# 6) Evaluate train/test
# ==============================
pred_train = xgboost_reg_weighted_bayes.predict(X_train_bo)
pred_test  = xgboost_reg_weighted_bayes.predict(X_test_bo)

if clip_pred_to_0_10:
    pred_train = np.clip(pred_train, 0, 10)
    pred_test  = np.clip(pred_test,  0, 10)

def metrics_block(y_true, y_pred, tol):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100.0)
    return mae, rmse, r2, acc

mae_tr, rmse_tr, r2_tr, acc_tr = metrics_block(y_train_bo, pred_train, tol)
mae_te, rmse_te, r2_te, acc_te = metrics_block(y_test_bo,  pred_test,  tol)

metrics_bayes = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", f"Accuracy@±{tol}"],
    "Train_bayes": [mae_tr, rmse_tr, r2_tr, acc_tr],
    "Test_bayes":  [mae_te, rmse_te, r2_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Higher"]
})

print("\nBayesian best model metrics")
display(metrics_bayes)


# ==============================
# 7) Predict example row + SHAP local
# ==============================
if "X_new_xgboost_reg_w" in globals():
    X_one = X_new_xgboost_reg_w.copy()
else:
    X_one = X_new_encoded_no_te.copy()

X_one = X_one.reindex(columns=X_train_bo.columns, fill_value=0).astype(float)

pred_example = float(xgboost_reg_weighted_bayes.predict(X_one)[0])
if clip_pred_to_0_10:
    pred_example = float(np.clip(pred_example, 0, 10))

print("\nExample prediction (bayes):", pred_example)

explainer = shap.TreeExplainer(xgboost_reg_weighted_bayes)

# Global SHAP on test sample
X_shap = X_test_bo.sample(min(300, len(X_test_bo)), random_state=42)
shap_values_global = explainer.shap_values(X_shap)

print("\nSHAP global (summary)")
shap.summary_plot(shap_values_global, X_shap, show=True)

print("\nSHAP global (bar)")
shap.summary_plot(shap_values_global, X_shap, plot_type="bar", show=True)

# Local SHAP for example
shap_values_one = explainer.shap_values(X_one)
base_value = explainer.expected_value
pred_raw = float(xgboost_reg_weighted_bayes.predict(X_one)[0])

print("\nSHAP local numbers")
print("E[f(X)] (baseline):", base_value)
print("f(X) (prediction raw):", pred_raw)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[0],
        base_values=base_value,
        data=X_one.iloc[0],
        feature_names=X_one.columns
    ),
    max_display=20
)
plt.show()


# ==============================
# 8) Save model + metadata
# ==============================
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = f"xgb_reg_weighted_optuna_{timestamp}"
save_dir = os.path.join("models", "xgboost", "regression", model_name)
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "model.json")
xgboost_reg_weighted_bayes.save_model(model_path)

meta = {
    "model_name": model_name,
    "created_at": timestamp,
    "model_type": "XGBRegressor",
    "task": "regression",
    "weighted_training": True,
    "optimization": {
        "method": "Optuna Bayesian (multi-objective)",
        "n_trials": n_trials,
        "cv": {"type": "KFold", "n_splits": n_splits, "shuffle": True, "random_state": random_state},
        "objectives": {
            "minimize": ["MAE_CV", "RMSE_CV"],
            "maximize": ["R2_CV", f"Accuracy@±{tol}_CV"]
        },
        "pareto_selection_score": {
            "formula": "score = -w_mae*MAE - w_rmse*RMSE + w_r2*R2 + w_acc*ACC",
            "weights": {"w_mae": w_mae, "w_rmse": w_rmse, "w_r2": w_r2, "w_acc": w_acc}
        },
        "best_trial_values": {
            "MAE_CV": float(best_trial.values[0]),
            "RMSE_CV": float(best_trial.values[1]),
            "R2_CV": float(best_trial.values[2]),
            f"Accuracy@±{tol}_CV": float(best_trial.values[3]),
        },
        "best_params": best_params
    },
    "test_metrics": {
        "MAE": mae_te,
        "RMSE": rmse_te,
        "R2": r2_te,
        f"Accuracy@±{tol}": acc_te
    },
    "feature_count": int(X_train_bo.shape[1]),
    "feature_columns": X_train_bo.columns.astype(str).tolist(),
}

meta_path = os.path.join(save_dir, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\nModel saved in:", save_dir)
print("Files:", model_path, meta_path)
```
