```
# ============================================================
# CatBoost — Random Search (weighted regression, imbalanced target)
# NO functions — clean blocks
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, ParameterSampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

import shap
import matplotlib.pyplot as plt


# ============================================================
# 0) Assumes you already have:
# - X_train, X_test, y_train, y_test
# - cat_cols (list of categorical column names)
# ============================================================

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)
print("Categorical cols:", cat_cols)


# ==============================
# 1) Build sample weights on y_train (inverse frequency bins)
# ==============================
n_bins = 8
clip_min, clip_max = 0.5, 5.0

try:
    y_bins = pd.qcut(y_train, q=min(n_bins, y_train.nunique()), duplicates="drop")
except Exception:
    y_bins = pd.cut(y_train, bins=min(n_bins, max(2, y_train.nunique())))

bin_counts = y_bins.value_counts()
weights_train = y_bins.map(lambda b: 1.0 / bin_counts[b]).astype(float).values
weights_train = weights_train / np.mean(weights_train)
weights_train = np.clip(weights_train, clip_min, clip_max)

print("\nWeights stats:", pd.Series(weights_train).describe())


# ==============================
# 2) K-Fold CV setup (used during random search)
# ==============================
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# ==============================
# 3) Random Search space (keep it reasonable)
#    (CatBoost key params)
# ==============================
param_space = {
    "iterations": [800, 1200, 1800, 2500],
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    "random_strength": [0.0, 0.5, 1.0, 2.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bylevel": [0.6, 0.8, 1.0],
}

n_iter = 25  # adjust 15..50 depending on your compute budget

random_params_list = list(ParameterSampler(param_space, n_iter=n_iter, random_state=42))
print(f"\nRandom Search trials: {len(random_params_list)}")


# ==============================
# 4) CV loop: evaluate each sampled config (weighted RMSE)
# ==============================
best_params = None
best_rmse = float("inf")
all_results = []

for trial_idx, params in enumerate(random_params_list, start=1):
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
    std_rmse = float(np.std(fold_rmses))

    row = {"trial": trial_idx, "rmse_mean": mean_rmse, "rmse_std": std_rmse, **params}
    all_results.append(row)

    if mean_rmse < best_rmse:
        best_rmse = mean_rmse
        best_params = params

    print(f"Trial {trial_idx:02d}/{len(random_params_list)} | RMSE={mean_rmse:.4f} ± {std_rmse:.4f}")

print("\n✅ Best CV RMSE:", best_rmse)
print("✅ Best params:", best_params)

results_df = pd.DataFrame(all_results).sort_values("rmse_mean").reset_index(drop=True)
display(results_df.head(10))


# ==============================
# 5) Train final best model on FULL train (with weights)
# ==============================
train_pool_full = Pool(X_train, y_train, cat_features=cat_cols, weight=weights_train)
test_pool       = Pool(X_test, y_test, cat_features=cat_cols)

cat_model_random_best = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False,
    **best_params
)

cat_model_random_best.fit(
    train_pool_full,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("\n✅ Final RandomSearch CatBoost trained")


# ==============================
# 6) Evaluate train/test + tolerance accuracy
# ==============================
pred_train = np.clip(cat_model_random_best.predict(X_train), 0, 10)
pred_test  = np.clip(cat_model_random_best.predict(X_test),  0, 10)

mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = r2_score(y_train, pred_train)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)

print("\n=== Random Search Best (Weighted) Metrics ===")
print(f"Train: MAE={mae_train:.4f} | RMSE={rmse_train:.4f} | R2={r2_train:.4f}")
print(f"Test : MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R2={r2_test:.4f}")

tol = 1.0
acc_train = (np.abs(y_train.values - pred_train) <= tol).mean() * 100
acc_test  = (np.abs(y_test.values  - pred_test)  <= tol).mean() * 100
print(f"\nAccuracy within ±{tol} point:")
print(f"Train: {acc_train:.2f}%")
print(f"Test : {acc_test:.2f}%")


# ==============================
# 7) Predict on your example row
# ==============================
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

# categorical to string + fill missing
for c in cat_cols:
    X_new[c] = X_new[c].astype("string").fillna("__MISSING__")

# numeric conversion
num_cols_new = [c for c in X_new.columns if c not in cat_cols]
for c in num_cols_new:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

pred_new = float(np.clip(cat_model_random_best.predict(X_new)[0], 0, 10))
print("\n=== Example prediction (RandomSearch best) ===")
print("Predicted note:", pred_new)


# ==============================
# 8) SHAP (global + example)
# ==============================
sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_model_random_best)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP: global feature importance")
shap.summary_plot(shap_values, X_shap, show=True)

print("\nSHAP: explanation for the example row")
shap_values_one = explainer.shap_values(X_new)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one[0],
        base_values=explainer.expected_value,
        data=X_new.iloc[0],
        feature_names=X_new.columns
    )
)
plt.show()

```
