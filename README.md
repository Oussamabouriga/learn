```
# ============================================================
# CATBOOST REGRESSION — TARGET WEIGHTS + BAYESIAN OPTIMIZATION (Optuna)
# - Uses your prepared CatBoost-ready data:
#     X_train_cb, X_test_cb, y_train_no_te, y_test_no_te, cat_cols_cb
# - Uses your prepared sample weights on TRAIN:
#     sample_weight_train_cat_w
# - Bayesian optimization with Optuna + KFold CV (weighted)
# - Objective: minimize CV RMSE (robust + standard for regression)
# - Train best model on full train (with weights)
# - Metrics + Accuracy@±tol
# - Predict example row (X_new_cb) + SHAP global + SHAP local
# - Save model in: models/catboost/regression/<model_name>/
#
# Requirements:
#   pip install catboost optuna shap scikit-learn
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error

import optuna
import shap


# ==============================
# 0) Inputs + checks
# ==============================
assert "X_train_cb" in globals() and "X_test_cb" in globals()
assert "y_train_no_te" in globals() and "y_test_no_te" in globals()
assert "cat_cols_cb" in globals()
assert "sample_weight_train_cat_w" in globals()

X_train_cat_bayes = X_train_cb.copy()
X_test_cat_bayes  = X_test_cb.copy()

y_train_cat_bayes = pd.to_numeric(pd.Series(y_train_no_te), errors="coerce").astype(float).values
y_test_cat_bayes  = pd.to_numeric(pd.Series(y_test_no_te),  errors="coerce").astype(float).values

w_train_cat_bayes = np.asarray(sample_weight_train_cat_w, dtype=float)

print("Train:", X_train_cat_bayes.shape, y_train_cat_bayes.shape, w_train_cat_bayes.shape)
print("Test :", X_test_cat_bayes.shape,  y_test_cat_bayes.shape)
print("Cat cols:", len(cat_cols_cb))


# ==============================
# 1) Config (editable)
# ==============================
random_state = 42
n_splits = 5
n_trials = 40          # 20..80 depending on compute
timeout_sec = None     # you can set e.g. 1800 to stop after 30 min
tol = 1.0              # Accuracy@±tol

cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ==============================
# 2) Optuna objective (weighted CV RMSE)
# ==============================
def objective(trial: optuna.Trial) -> float:
    params = {
        # fixed (safe defaults)
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,

        # tunable
        "iterations": trial.suggest_int("iterations", 600, 2600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),

        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),

        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),

        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
    }

    fold_rmses = []

    for tr_idx, va_idx in cv.split(X_train_cat_bayes):
        X_tr = X_train_cat_bayes.iloc[tr_idx]
        y_tr = y_train_cat_bayes[tr_idx]
        w_tr = w_train_cat_bayes[tr_idx]

        X_va = X_train_cat_bayes.iloc[va_idx]
        y_va = y_train_cat_bayes[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols_cb, weight=w_tr)
        val_pool   = Pool(X_va, y_va, cat_features=cat_cols_cb)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=200)

        pred_va = np.clip(model.predict(X_va), 0, 10)
        rmse_va = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        fold_rmses.append(rmse_va)

    return float(np.mean(fold_rmses))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials, timeout=timeout_sec)

best_params_cat_bayes = study.best_params
best_cv_rmse_cat_bayes = float(study.best_value)

print("\nBest CV RMSE:", best_cv_rmse_cat_bayes)
print("Best params:", best_params_cat_bayes)


# ==============================
# 3) Train final best model (weights) + early stopping
# ==============================
model_name = "catboost_reg_weighted_optuna_no_te_v1"

final_cat_bayes = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=random_state,
    verbose=200,
    allow_writing_files=False,
    **best_params_cat_bayes
)

train_pool_full = Pool(
    X_train_cat_bayes, y_train_cat_bayes,
    cat_features=cat_cols_cb,
    weight=w_train_cat_bayes
)
test_pool = Pool(
    X_test_cat_bayes, y_test_cat_bayes,
    cat_features=cat_cols_cb
)

final_cat_bayes.fit(
    train_pool_full,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("Trained final model:", model_name)


# ==============================
# 4) Metrics + Accuracy@±tol
# ==============================
pred_train = np.clip(final_cat_bayes.predict(X_train_cat_bayes), 0, 10)
pred_test  = np.clip(final_cat_bayes.predict(X_test_cat_bayes),  0, 10)

def _metrics(y_true, y_pred, tol):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100)
    return mae, rmse, r2, medae, mxerr, acc

mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr = _metrics(y_train_cat_bayes, pred_train, tol)
mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te = _metrics(y_test_cat_bayes,  pred_test,  tol)

metrics_cat_bayes = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", f"Accuracy@±{tol}"],
    "Train":  [mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr],
    "Test":   [mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher"]
})

print("\n=== CATBOOST (Weighted + Optuna) Metrics ===")
display(metrics_cat_bayes)


# ==============================
# 5) Example prediction + SHAP
# ==============================
pred_example = None
if "X_new_cb" in globals():
    pred_example = float(np.clip(final_cat_bayes.predict(X_new_cb)[0], 0, 10))
    print("\nExample prediction:", pred_example)
else:
    print("\nX_new_cb not found — skipping example prediction.")

sample_size = min(300, len(X_test_cat_bayes))
X_shap = X_test_cat_bayes.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(final_cat_bayes)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP global (summary)")
shap.summary_plot(shap_values, X_shap, show=True)
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=True)

if "X_new_cb" in globals():
    shap_one = explainer.shap_values(X_new_cb)
    base_val = explainer.expected_value
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_one[0],
            base_values=base_val,
            data=X_new_cb.iloc[0],
            feature_names=X_new_cb.columns
        ),
        max_display=20
    )
    plt.show()


# ==============================
# 6) Save model + artifacts
# ==============================
save_dir = os.path.join("models", "catboost", "regression", model_name)
os.makedirs(save_dir, exist_ok=True)

final_cat_bayes.save_model(os.path.join(save_dir, "model.cbm"))
metrics_cat_bayes.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params_cat_bayes, f, indent=2)

meta = {
    "model_name": model_name,
    "best_cv_rmse": best_cv_rmse_cat_bayes,
    "tol": tol,
    "example_prediction": pred_example,
    "n_trials": n_trials,
    "n_splits": n_splits
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Saved to:", save_dir)

```
