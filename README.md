```
# ============================================================
# CatBoost Regressor (baseline) — FULL WORKING NOTEBOOK BLOCK
# - Uses categorical columns directly (NO one-hot needed)
# - Forces code_postal as categorical
# - Fixes pd.NA / <NA> issue (convert to np.nan) -> CatBoost friendly
# - Trains CatBoostRegressor
# - Evaluates: MAE, RMSE, R2 + "accuracy within tolerance"
# - Predicts on your example row + SHAP global + SHAP row explanation
# ============================================================

# ==============================
# 0) Imports
# ==============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Inputs you must have
# ==============================
# df must exist
target_col = "evaluate_note"

feature_cols = [c for c in df.columns if c != target_col]
force_categorical_cols = ["code_postal"]  # add others if needed


# ==============================
# 2) Build X / y
# ==============================
X = df[feature_cols].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

mask_y = y.notna()
X = X.loc[mask_y].copy()
y = y.loc[mask_y].astype(float).copy()


# ==============================
# 3) Force selected columns as categorical (string)
# ==============================
for c in force_categorical_cols:
    if c in X.columns:
        X[c] = X[c].astype("string")


# ==============================
# 4) Train / test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)


# ==============================
# 5) Fix pd.NA / <NA> for CatBoost
#    CatBoost accepts np.nan, NOT pd.NA (NAType)
# ==============================
def _fix_na_for_catboost(df_):
    df_ = df_.copy()

    # Convert pandas missing values to real np.nan
    df_ = df_.replace({pd.NA: np.nan})

    # Ensure numeric columns are numeric (NaN allowed)
    num_cols = df_.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df_[c] = pd.to_numeric(df_[c], errors="coerce")

    # Final safety: ensure remaining missing are np.nan
    # (keeps strings/objects, just replaces missing)
    df_ = df_.where(pd.notna(df_), np.nan)

    return df_

X_train = _fix_na_for_catboost(X_train)
X_test  = _fix_na_for_catboost(X_test)


# ==============================
# 6) Verify no pd.NA left (NO applymap)
# ==============================
# This is safe and fast: checks the underlying numpy array
has_pdna_train = np.any(X_train.to_numpy(dtype=object) == pd.NA)
has_pdna_test  = np.any(X_test.to_numpy(dtype=object) == pd.NA)

print("Any pd.NA left in train?", has_pdna_train)
print("Any pd.NA left in test ?", has_pdna_test)


# ==============================
# 7) Detect categorical columns for CatBoost
# ==============================
cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

# Ensure forced columns are included
for c in force_categorical_cols:
    if c in X_train.columns and c not in cat_cols:
        cat_cols.append(c)

cat_features = cat_cols

print("\nCategorical columns used by CatBoost:")
print(cat_features)


# ==============================
# 8) Build Pools
# ==============================
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool  = Pool(X_test, y_test, cat_features=cat_features)


# ==============================
# 9) Baseline CatBoost Regressor
# ==============================
cat_model_baseline = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=42,
    eval_metric="RMSE",
    verbose=200,
    allow_writing_files=False
)

cat_model_baseline.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("\nCatBoost baseline trained ✅")


# ==============================
# 10) Predictions + metrics
# ==============================
pred_train = cat_model_baseline.predict(X_train)
pred_test  = cat_model_baseline.predict(X_test)

pred_train_clipped = np.clip(pred_train, 0, 10)
pred_test_clipped  = np.clip(pred_test, 0, 10)

mae_train = mean_absolute_error(y_train, pred_train_clipped)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train_clipped))
r2_train = r2_score(y_train, pred_train_clipped)

mae_test = mean_absolute_error(y_test, pred_test_clipped)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test_clipped))
r2_test = r2_score(y_test, pred_test_clipped)

print("\n=== CatBoost Baseline Metrics ===")
print(f"Train: MAE={mae_train:.4f} | RMSE={rmse_train:.4f} | R2={r2_train:.4f}")
print(f"Test : MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R2={r2_test:.4f}")

# Accuracy within tolerance (like you used before)
tol = 1.0
acc_train_tol = (np.abs(y_train.values - pred_train_clipped) <= tol).mean() * 100
acc_test_tol  = (np.abs(y_test.values  - pred_test_clipped)  <= tol).mean() * 100
print(f"\nAccuracy within ±{tol} point:")
print(f"Train: {acc_train_tol:.2f}%")
print(f"Test : {acc_test_tol:.2f}%")


# ==============================
# 11) Predict on your example row (same columns order)
# ==============================
new_rows = [
    {
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
    }
]

X_new = pd.DataFrame(new_rows)

# add missing columns if any
missing_cols = [c for c in X_train.columns if c not in X_new.columns]
for c in missing_cols:
    X_new[c] = np.nan

# align order
X_new = X_new[X_train.columns]

# force code_postal categorical
if "code_postal" in X_new.columns:
    X_new["code_postal"] = X_new["code_postal"].astype("string")

X_new = _fix_na_for_catboost(X_new)

pred_new = cat_model_baseline.predict(X_new)[0]
pred_new = float(np.clip(pred_new, 0, 10))

print("\n=== Prediction on your example row ===")
print("Predicted note:", pred_new)


# ==============================
# 12) SHAP (global + example)
# ==============================
# Use a small sample for speed
sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()
X_shap = _fix_na_for_catboost(X_shap)

explainer = shap.TreeExplainer(cat_model_baseline)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP: Global importance (summary plot)")
shap.summary_plot(shap_values, X_shap, show=True)

print("\nSHAP: Explanation for your single example row")
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

print("\n✅ Done (CatBoost + metrics + example prediction + SHAP)")

```
