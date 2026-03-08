```
# ============================================================
# CATBOOST REGRESSION — TARGET WEIGHTS + SMALL GRID SEARCH (CV)
# - Uses your prepared CatBoost-ready data:
#     X_train_cb, X_test_cb, y_train_no_te, y_test_no_te, cat_cols_cb
# - Uses your prepared sample weights on TRAIN:
#     sample_weight_train_cat_w
# - GridSearchCV (small / compute-friendly)
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
from sklearn.model_selection import GridSearchCV, KFold
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

X_train_cat_gs = X_train_cb.copy()
X_test_cat_gs  = X_test_cb.copy()

y_train_cat_gs = pd.to_numeric(pd.Series(y_train_no_te), errors="coerce").astype(float).values
y_test_cat_gs  = pd.to_numeric(pd.Series(y_test_no_te),  errors="coerce").astype(float).values

w_train_cat_gs = np.asarray(sample_weight_train_cat_w, dtype=float)

print("Train:", X_train_cat_gs.shape, y_train_cat_gs.shape, w_train_cat_gs.shape)
print("Test :", X_test_cat_gs.shape,  y_test_cat_gs.shape)
print("Cat cols:", len(cat_cols_cb))


# ==============================
# 1) Base model + CV
# ==============================
base_cat_gs = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# ==============================
# 2) SMALL Grid (compute-friendly)
# Keep it small: 2-4 values per param max
# ==============================
param_grid = {
    "iterations": [800, 1500],
    "learning_rate": [0.03, 0.07],
    "depth": [6, 8],
    "l2_leaf_reg": [3.0, 10.0],
    "subsample": [0.8, 1.0],
    "colsample_bylevel": [0.8, 1.0],
    "grow_policy": ["SymmetricTree"]  # keep 1 choice to limit compute
}

grid_search_cat = GridSearchCV(
    estimator=base_cat_gs,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    verbose=2,
    n_jobs=1,     # CatBoost parallelizes internally
    refit=True,
    return_train_score=True
)


# ==============================
# 3) Fit Grid Search (with weights + cat columns)
# ==============================
grid_search_cat.fit(
    X_train_cat_gs,
    y_train_cat_gs,
    cat_features=cat_cols_cb,
    sample_weight=w_train_cat_gs
)

best_params_cat_gs = grid_search_cat.best_params_
best_cv_rmse_cat_gs = float(-grid_search_cat.best_score_)

print("\nBest CV RMSE:", best_cv_rmse_cat_gs)
print("Best params:", best_params_cat_gs)


# ==============================
# 4) Train final best model (weights) + early stopping
# ==============================
model_name = "catboost_reg_weighted_gridsearch_no_te_v1"

final_cat_gs = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    **best_params_cat_gs
)

train_pool_gs = Pool(
    X_train_cat_gs, y_train_cat_gs,
    cat_features=cat_cols_cb,
    weight=w_train_cat_gs
)
test_pool_gs = Pool(
    X_test_cat_gs, y_test_cat_gs,
    cat_features=cat_cols_cb
)

final_cat_gs.fit(
    train_pool_gs,
    eval_set=test_pool_gs,
    use_best_model=True,
    early_stopping_rounds=200
)

print("Trained final model:", model_name)


# ==============================
# 5) Predict + metrics + Accuracy@±tol
# ==============================
tol = 1.0  # change to 0.5 if you want

pred_train = np.clip(final_cat_gs.predict(X_train_cat_gs), 0, 10)
pred_test  = np.clip(final_cat_gs.predict(X_test_cat_gs),  0, 10)

def _metrics(y_true, y_pred, tol):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    acc = float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= tol).mean() * 100)
    return mae, rmse, r2, medae, mxerr, acc

mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr = _metrics(y_train_cat_gs, pred_train, tol)
mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te = _metrics(y_test_cat_gs,  pred_test,  tol)

metrics_cat_gs = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", f"Accuracy@±{tol}"],
    "Train":  [mae_tr, rmse_tr, r2_tr, medae_tr, mxerr_tr, acc_tr],
    "Test":   [mae_te, rmse_te, r2_te, medae_te, mxerr_te, acc_te],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher"]
})

print("\n=== CATBOOST (Weighted + Grid Search) Metrics ===")
display(metrics_cat_gs)


# ==============================
# 6) Example prediction + SHAP
# ==============================
pred_example = None
if "X_new_cb" in globals():
    pred_example = float(np.clip(final_cat_gs.predict(X_new_cb)[0], 0, 10))
    print("\nExample prediction:", pred_example)
else:
    print("\nX_new_cb not found — skipping example prediction.")

sample_size = min(300, len(X_test_cat_gs))
X_shap = X_test_cat_gs.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(final_cat_gs)
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
# 7) Save model + artifacts
# ==============================
save_dir = os.path.join("models", "catboost", "regression", model_name)
os.makedirs(save_dir, exist_ok=True)

final_cat_gs.save_model(os.path.join(save_dir, "model.cbm"))
metrics_cat_gs.to_csv(os.path.join(save_dir, "metrics_train_test.csv"), index=False)

with open(os.path.join(save_dir, "best_params.json"), "w") as f:
    json.dump(best_params_cat_gs, f, indent=2)

meta = {
    "model_name": model_name,
    "best_cv_rmse": best_cv_rmse_cat_gs,
    "tol": tol,
    "example_prediction": pred_example,
    "grid_size_total": int(np.prod([len(v) for v in param_grid.values()]))
}
with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Saved to:", save_dir)

```
