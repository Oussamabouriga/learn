```

# ============================================================
# CatBoost Regressor (baseline) — WORKING BLOCK (NO applymap)
# Fixes:
# - pd.NA -> np.nan everywhere
# - explicit categorical columns (so numeric columns don't get mis-detected)
# - categorical missing -> "__MISSING__" (safe for CatBoost)
# - numeric columns -> pd.to_numeric(errors="coerce")
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor, Pool

import shap
import matplotlib.pyplot as plt


# ==============================
# 1) Define target + categorical columns (EXPLICIT)
# ==============================
target_col = "evaluate_note"

cat_cols = [
    "PARCOURS_FINAL",
    "PARCOURS_INITIAL",
    "operating_system",
    "marque",
    "model",
    "garantie",
    "list_prest",
    "code_postal",          # forced categorical
]

# Keep only columns that exist in df
cat_cols = [c for c in cat_cols if c in df.columns]


# ==============================
# 2) Build X / y
# ==============================
feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].astype(float).copy()


# ==============================
# 3) Convert pd.NA -> np.nan (VERY IMPORTANT)
# ==============================
X = X.replace({pd.NA: np.nan})


# ==============================
# 4) Force categorical cols to string + fill missing
# ==============================
for c in cat_cols:
    X[c] = X[c].astype("string")
    X[c] = X[c].fillna("__MISSING__")   # important: avoid pd.NA / NaN inside cat cols


# ==============================
# 5) Convert ALL non-categorical columns to numeric (safe)
#    (This prevents numeric columns like montant_indem becoming "object" and treated as cat)
# ==============================
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")   # keeps np.nan


# ==============================
# 6) Train/test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)


# ==============================
# 7) Check missing safely (NO pd.NA comparison!)
# ==============================
print("\nTop missing columns (train):")
print(X_train.isna().sum().sort_values(ascending=False).head(10))

print("\nTop missing columns (test):")
print(X_test.isna().sum().sort_values(ascending=False).head(10))

# Extra: verify categorical cols have no missing (we filled them)
print("\nAny missing inside categorical columns (train)?",
      X_train[cat_cols].isna().to_numpy().any())
print("Any missing inside categorical columns (test)?",
      X_test[cat_cols].isna().to_numpy().any())

print("\nCategorical cols used:", cat_cols)


# ==============================
# 8) Build CatBoost Pools
# ==============================
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_cols)


# ==============================
# 9) Train baseline CatBoost Regressor
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
# 10) Metrics + tolerance accuracy
# ==============================
pred_train = np.clip(cat_model_baseline.predict(X_train), 0, 10)
pred_test  = np.clip(cat_model_baseline.predict(X_test),  0, 10)

mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = r2_score(y_train, pred_train)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)

print("\n=== CatBoost Baseline Metrics ===")
print(f"Train: MAE={mae_train:.4f} | RMSE={rmse_train:.4f} | R2={r2_train:.4f}")
print(f"Test : MAE={mae_test:.4f} | RMSE={rmse_test:.4f} | R2={r2_test:.4f}")

tol = 1.0
acc_train = (np.abs(y_train.values - pred_train) <= tol).mean() * 100
acc_test  = (np.abs(y_test.values  - pred_test)  <= tol).mean() * 100
print(f"\nAccuracy within ±{tol} point:")
print(f"Train: {acc_train:.2f}%")
print(f"Test : {acc_test:.2f}%")


# ==============================
# 11) Predict on your example row (and align columns)
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

# align order
X_new = X_new[X_train.columns].copy()

# pd.NA -> np.nan
X_new = X_new.replace({pd.NA: np.nan})

# apply same transformations
for c in cat_cols:
    X_new[c] = X_new[c].astype("string").fillna("__MISSING__")

num_cols = [c for c in X_new.columns if c not in cat_cols]
for c in num_cols:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

pred_new = float(np.clip(cat_model_baseline.predict(X_new)[0], 0, 10))
print("\n=== Example prediction ===")
print("Predicted note:", pred_new)


# ==============================
# 12) SHAP (global + example)
# ==============================
# Use a sample for speed
sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()

explainer = shap.TreeExplainer(cat_model_baseline)
shap_values = explainer.shap_values(X_shap)

print("\nSHAP: global importance")
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
