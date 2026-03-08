```

# ============================================================
# CATBOOST REGRESSION — TARGET WEIGHTS + RANDOM SEARCH (CV)
# - Uses your prepared CatBoost-ready data:
#     X_train_cb, X_test_cb, y_train_no_te, y_test_no_te, cat_cols_cb
# - Uses your prepared sample weights on TRAIN:
#     sample_weight_train_cat_w
# - RandomizedSearchCV over CatBoost hyperparams (compute-friendly)
# - Refit best model on full train (with weights)
# - Metrics + Accuracy@±tol
# - Predict example row (X_new_cb) + SHAP global + SHAP local
# - Save model in: models/catboost/regression/<model_name>/
#
# Requirements:
#   pip install catboost shap scikit-learn
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error
)
import shap


# ==============================
# 0) Safety checks
# ==============================
assert "X_train_cb" in globals() and "X_test_cb" in globals()
assert "y_train_no_te" in globals() and "y_test_no_te" in globals()
assert "cat_cols_cb" in globals()
assert "sample_weight_train_cat_w" in globals()

X_train_cat_rs = X_train_cb.copy()
X_test_cat_rs  = X_test_cb.copy()

y_train_cat_rs = pd.to_numeric(pd.Series(y_train_no_te), errors="coerce").astype(float).values
y_test_cat_rs  = pd.to_numeric(pd.Series(y_test_no_te),  errors="coerce").astype(float).values

w_train_cat_rs = np.asarray(sample_weight_train_cat_w, dtype=float)

print("Train:", X_train_cat_rs.shape, y_train_cat_rs.shape, w_train_cat_rs.shape)
print("Test :", X_test_cat_rs.shape,  y_test_cat_rs.shape)
print("Cat cols:", len(cat_cols_cb))


# ==============================
# 1) Base model (fixed params) + CV
# ==============================
base_cat_rs = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ==============================
# 2) Random Search space (focused)
#    (keep it reasonable: CatBoost is expensive)
# ==============================
param_distributions = {
    "iterations": [600, 1000, 1500, 2000, 2500],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    "depth": [4, 5, 6, 7, 8, 9, 10],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
    "random_strength": [0.0, 0.5, 1.0, 1.5, 2.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1.0],
    "grow_policy": ["SymmetricTree", "Lossguide"]
}

n_iter = 30  # adjust 20..60 depending on compute budget

random_search_cat = RandomizedSearchCV(
    estimator=base_cat_rs,
    param_distributions=param_distributions,
    n_iter=n_iter,
    scoring="neg_root_mean_squared_error",  # optimize RMSE (lower is better)
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=1,  # CatBoost parallelizes internally; keep this 1 to avoid overload
    refit=True,
    return_train_score=True
)


# ==============================
# 3) Fit Random Search (with weights + categorical columns)
# IMPORTANT:
#   - pass categorical features via fit params
#   - pass sample_weight to handle imbalanced regression
# ==============================
random_search_cat.fit(
    X_train_cat_rs,
    y_train_cat_rs,
    cat_features=cat_cols_cb,
    sample_weight=w_train_cat_rs
)

best_params_cat_rs = random_search_cat.best_params_
best_cv_rmse_cat_rs = float(-random_search_cat.best_score_)

print("\nBest CV RMSE:", best_cv_rmse_cat_rs)
print("Best params:", best_params_cat_rs)


# ==============================
# 4) Train final best model on full train (weights) with early stopping
#    (RandomizedSearchCV already refit, but we do a clean final fit for logs)
# ==============================
model_name = "catboost_reg_weighted_randomsearch_no_te_v1"

final_cat_rs = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    **best_params_cat_rs
)

train_pool_rs = Pool(
    X_train_cat_rs, y_train_cat_rs,
    cat_features=cat_cols_cb,
    weight=w_train_cat_rs
)
test_pool_rs = Pool(
    X_test_cat_rs, y_test_cat_rs,
    cat_features=cat_cols_cb
)

final_cat_rs.fit(
    train_pool_rs,
    eval_set=test_pool_rs,
    use_best_model=True,
    early_stopping_rounds=200
)

print("Trained final model:", model_name)


# ==============================
# 5) Predict + metrics + Accuracy@±tol
# ==============================
tol = 1.0  # change to 0.5 if you want

pred_train = np.clip(final_cat_rs.predict(X_train_cat_rs), 0, 10)
pred_test  = np.clip(final_cat_rs.predict(X_test_cat_rs),  0, 10)

def _metrics(y_true, y_pred, tol):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100)
    return mae, rmse, r2, medae, mxerr, acc

mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr = _metrics(y_train_cat_rs, pred_train, tol)
mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te = _metrics(y_test_cat_rs,  pred_test,  tol)

metrics_cat_rs = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", f"Accuracy@±{tol}"],
    "Train":  [mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr],
    "Test":   [mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher"]
})

print("\n=== CATBOOST (Weighted + Random Search) Metrics ===")
display(metrics_cat_rs)


# ==============================
# 6) Predict your example row + SHAP local
# ==============================
pred_example = None
if "X_new_cb" in globals():
    pred_example = float(np.clip(final_cat_rs.predict(X_new_cb)[0], 0, 10))
    print("\nExample prediction:", pred_example)
else:
    print("\nX_new_cb not found — skipping example prediction.")


# ==============================
# 7) SHAP: global + example
# Notes:
# - CatBoost works with TreeExplainer
# - Use a sample from test for global
# ==============================
sample_size = min(300, len(X_test_cat_rs))
X_shap = X_test_cat_rs.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(final_cat_rs)
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
# 8) Save model + artifacts
# ==============================
save_dir = os.path.join("models", "catboost", "regression", model_name)
os.makedirs(save_dir, exist_ok=True)

final_cat_rs.save_model(os.path.join(save_dir, "model.cbm"))
metrics_cat_rs.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params_cat_rs, f, indent=2)

meta = {
    "model_name": model_name,
    "n_iter_random_search": n_iter,
    "best_cv_rmse": best_cv_rmse_cat_rs,
    "tol": tol,
    "example_prediction": pred_example
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Saved to:", save_dir)
```
