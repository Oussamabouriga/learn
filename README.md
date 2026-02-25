```
# ============================================================
# CatBoost — SMALL Grid Search (compute-friendly, weighted regression)
# Starts from your best random-search params and tests only a small neighborhood
# NO functions — clean blocks
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

import shap
import matplotlib.pyplot as plt


# ============================================================
# 0) Assumes you already have (from previous steps):
# - X_train, X_test, y_train, y_test
# - cat_cols (categorical column names)
# - weights_train (sample weights computed from y_train)
# - best_params (from Random Search)
# ============================================================

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
print("cat_cols:", cat_cols)
print("weights_train:", pd.Series(weights_train).describe())

# KFold used by Grid Search (same idea as before)
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 1) Build a small grid around best_params
#    (VERY IMPORTANT: keep it small to be compute-friendly)
# ============================================================

bp = dict(best_params)  # best params from random search

# Small neighborhood (only a few values each)
grid_space = {
    # iterations close to the best
    "iterations": sorted(list(set([
        bp.get("iterations", 1500),
        int(bp.get("iterations", 1500) * 0.75),
        int(bp.get("iterations", 1500) * 1.10),
    ]))),

    # learning rate close to the best
    "learning_rate": sorted(list(set([
        bp.get("learning_rate", 0.03),
        max(0.005, bp.get("learning_rate", 0.03) * 0.7),
        min(0.2,  bp.get("learning_rate", 0.03) * 1.3),
    ]))),

    # depth close to the best (±1)
    "depth": sorted(list(set([
        bp.get("depth", 6),
        max(2, bp.get("depth", 6) - 1),
        bp.get("depth", 6) + 1,
    ]))),

    # regularization close to the best
    "l2_leaf_reg": sorted(list(set([
        bp.get("l2_leaf_reg", 3.0),
        max(0.1, bp.get("l2_leaf_reg", 3.0) * 0.7),
        bp.get("l2_leaf_reg", 3.0) * 1.3,
    ]))),

    # keep these small (2 values each)
    "random_strength": sorted(list(set([
        bp.get("random_strength", 1.0),
        bp.get("random_strength", 1.0) + 0.5,
    ]))),

    "bagging_temperature": sorted(list(set([
        bp.get("bagging_temperature", 1.0),
        bp.get("bagging_temperature", 1.0) + 0.5,
    ]))),

    "subsample": sorted(list(set([
        bp.get("subsample", 0.8),
        min(1.0, bp.get("subsample", 0.8) + 0.1),
    ]))),

    "colsample_bylevel": sorted(list(set([
        bp.get("colsample_bylevel", 0.8),
        min(1.0, bp.get("colsample_bylevel", 0.8) + 0.1),
    ]))),
}

grid_list = list(ParameterGrid(grid_space))
print(f"\nSmall Grid size: {len(grid_list)} configs")
print("Grid preview:", grid_list[:2])


# ============================================================
# 2) Grid Search CV (weighted) — optimize RMSE
# ============================================================

best_grid_params = None
best_grid_rmse = float("inf")
grid_results = []

for cfg_idx, params in enumerate(grid_list, start=1):
    fold_rmses = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train), start=1):
        X_tr, X_va = X_train.iloc[tr_idx].copy(), X_train.iloc[va_idx].copy()
        y_tr, y_va = y_train.iloc[tr_idx].copy(), y_train.iloc[va_idx].copy()

        w_tr = weights_train[tr_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, weight=w_tr)
        val_pool   = Pool(X_va, y_va, cat_features=cat_cols)

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            **params
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=200
        )

        pred_va = np.clip(model.predict(X_va), 0, 10)
        rmse_va = float(np.sqrt(mean_squared_error(y_va, pred_va)))
        fold_rmses.append(rmse_va)

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse  = float(np.std(fold_rmses))

    row = {"grid_cfg": cfg_idx, "rmse_mean": mean_rmse, "rmse_std": std_rmse, **params}
    grid_results.append(row)

    if mean_rmse < best_grid_rmse:
        best_grid_rmse = mean_rmse
        best_grid_params = params

    print(f"Grid {cfg_idx:02d}/{len(grid_list)} | RMSE={mean_rmse:.4f} ± {std_rmse:.4f}")

print("\n✅ Best GRID CV RMSE:", best_grid_rmse)
print("✅ Best GRID params:", best_grid_params)

grid_results_df = pd.DataFrame(grid_results).sort_values("rmse_mean").reset_index(drop=True)
display(grid_results_df.head(10))


# ============================================================
# 3) Train final best GRID model on FULL train (with weights)
# ============================================================

train_pool_full = Pool(X_train, y_train, cat_features=cat_cols, weight=weights_train)
test_pool       = Pool(X_test,  y_test,  cat_features=cat_cols)

cat_model_grid_best = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    **best_grid_params
)

cat_model_grid_best.fit(
    train_pool_full,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("\n✅ Final GRID CatBoost trained")


# ============================================================
# 4) Evaluate train/test + tolerance accuracy
# ============================================================

pred_train_g = np.clip(cat_model_grid_best.predict(X_train), 0, 10)
pred_test_g  = np.clip(cat_model_grid_best.predict(X_test),  0, 10)

mae_train_g  = mean_absolute_error(y_train, pred_train_g)
rmse_train_g = np.sqrt(mean_squared_error(y_train, pred_train_g))
r2_train_g   = r2_score(y_train, pred_train_g)

mae_test_g   = mean_absolute_error(y_test, pred_test_g)
rmse_test_g  = np.sqrt(mean_squared_error(y_test, pred_test_g))
r2_test_g    = r2_score(y_test, pred_test_g)

print("\n=== Small Grid Best (Weighted) Metrics ===")
print(f"Train: MAE={mae_train_g:.4f} | RMSE={rmse_train_g:.4f} | R2={r2_train_g:.4f}")
print(f"Test : MAE={mae_test_g:.4f} | RMSE={rmse_test_g:.4f} | R2={r2_test_g:.4f}")

tol = 1.0
acc_train_g = (np.abs(y_train.values - pred_train_g) <= tol).mean() * 100
acc_test_g  = (np.abs(y_test.values  - pred_test_g)  <= tol).mean() * 100
print(f"\nAccuracy within ±{tol} point:")
print(f"Train: {acc_train_g:.2f}%")
print(f"Test : {acc_test_g:.2f}%")


# ============================================================
# 5) Predict on the same example row (same as before)
# ============================================================

new_rows = [{
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "tarif": 19.99,
    "Nombre_sisnitre_client": 1,
    "Nombre_sisnitre_accepte_client": 1,
    "Nombre_sisnitre_refuse_client": np.nan,
    "Nombre_sisnitre_sans_suite_client": np.nan,
    "code_postal": 59700,
    "operating_system": "Android",
    "marque": "Google",
    "model": "Pixel 7 Pro ",
    "ancienneté_de_contrat": 509555,
    "garantie": "Dommage",
    "Age": 43,
    "dossier_complet": 1,
    "decision_ai": 0,
    "nombre_prestation_ko": 0,
    "Nbr_ticket_pieces": 0,
    "Nbr_ticket_information": 4,
    "list_prest": "ADVANCED_SWAP",
    "delai_declaration": 279000,
    "delai_de_completude": np.nan,
    "delai_decision": 13090,
    "delai_reparation": 4,
    "delai_indemnisation": 4,
    "montant_indem": np.nan,
    "delai_Sinistre": 602000,
}]

X_new = pd.DataFrame(new_rows)

# add missing cols
for c in X_train.columns:
    if c not in X_new.columns:
        X_new[c] = np.nan
X_new = X_new[X_train.columns].copy()

# pd.NA -> np.nan
X_new = X_new.replace({pd.NA: np.nan})

# categorical as string + fill missing
for c in cat_cols:
    X_new[c] = X_new[c].astype("string").fillna("__MISSING__")

# numeric columns to numeric
num_cols_new = [c for c in X_new.columns if c not in cat_cols]
for c in num_cols_new:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

pred_new_g = float(np.clip(cat_model_grid_best.predict(X_new)[0], 0, 10))
print("\n=== Example prediction (Grid best) ===")
print("Predicted note:", pred_new_g)


# ============================================================
# 6) SHAP (global + example row) for the GRID best model
# ============================================================

sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()

explainer_g = shap.TreeExplainer(cat_model_grid_best)
shap_values_g = explainer_g.shap_values(X_shap)

print("\nSHAP: global feature importance (GRID best)")
shap.summary_plot(shap_values_g, X_shap, show=True)

print("\nSHAP: explanation for the example row (GRID best)")
shap_values_one_g = explainer_g.shap_values(X_new)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one_g[0],
        base_values=explainer_g.expected_value,
        data=X_new.iloc[0],
        feature_names=X_new.columns
    )
)
plt.show()

```
