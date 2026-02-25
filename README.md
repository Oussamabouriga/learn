```

# ============================================================
# CATBOOST REGRESSOR — BASELINE (Modèle 1) + SHAP + EXAMPLE ROW
# - Force "code_postal" as categorical
# - Train/Test metrics (MAE, RMSE, R2 + Accuracy@tol)
# - Predict on ONE example row
# - SHAP global importance + SHAP local explanation for the example
# ============================================================

# ------------------------------
# 1) Install / Imports
# ------------------------------
# If needed once:
# !pip install catboost shap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    max_error,
    explained_variance_score
)

import shap

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)


# ------------------------------
# 2) Config (editable)
# ------------------------------
target_col = "evaluate_note"
test_size = 0.2
random_state = 42

# Optional: if 0 means "missing" in these columns, convert 0 -> np.nan
zero_to_nan_cols = [
    # "delai_de_completude",
    # "montant_indem",
]

# Business range for note
clip_min, clip_max = 0, 10

# Accuracy tolerance (in points)
tol = 1.0  # change to 0.5 if needed


# ------------------------------
# 3) Prepare dataframe (NO encoding needed)
# ------------------------------
df_cb = df.copy()
df_cb.columns = df_cb.columns.astype(str).str.strip()

# Ensure target is numeric and remove missing targets
df_cb[target_col] = pd.to_numeric(df_cb[target_col], errors="coerce")
df_cb = df_cb[df_cb[target_col].notna()].copy()

# Optional: convert selected 0 -> np.nan
for c in zero_to_nan_cols:
    if c in df_cb.columns:
        df_cb[c] = pd.to_numeric(df_cb[c], errors="coerce")
        df_cb.loc[df_cb[c] == 0, c] = np.nan

# Split X/y
X = df_cb.drop(columns=[target_col]).copy()
y = df_cb[target_col].astype(float).copy()


# ------------------------------
# 4) Force "code_postal" as categorical
# ------------------------------
if "code_postal" in X.columns:
    X["code_postal"] = X["code_postal"].astype("string")

# Detect categorical columns for CatBoost
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
print("Categorical columns used by CatBoost:", cat_cols)


# ------------------------------
# 5) Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)


# ------------------------------
# 6) Build Pools (important for CatBoost)
# ------------------------------
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool  = Pool(X_test, y_test, cat_features=cat_cols)


# ------------------------------
# 7) Baseline CatBoost model
# ------------------------------
cat_model_baseline = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=random_state,

    iterations=2000,
    learning_rate=0.03,
    depth=6,

    l2_leaf_reg=3.0,
    random_strength=1.0,

    bootstrap_type="Bernoulli",
    subsample=0.8,

    verbose=200
)

cat_model_baseline.fit(train_pool)
print("CatBoost baseline trained ✅")


# ------------------------------
# 8) Predict (train + test)
# ------------------------------
pred_train = cat_model_baseline.predict(train_pool)
pred_test  = cat_model_baseline.predict(test_pool)

# Clip to business range
pred_train = np.clip(pred_train, clip_min, clip_max)
pred_test  = np.clip(pred_test, clip_min, clip_max)

print("Prediction done ✅")


# ------------------------------
# 9) Evaluate (many metrics)
# ------------------------------
mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = r2_score(y_train, pred_train)
medae_train = median_absolute_error(y_train, pred_train)
maxerr_train = max_error(y_train, pred_train)
evs_train = explained_variance_score(y_train, pred_train)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)
medae_test = median_absolute_error(y_test, pred_test)
maxerr_test = max_error(y_test, pred_test)
evs_test = explained_variance_score(y_test, pred_test)

# "Accuracy" regression: % within tolerance
acc_train = (np.abs(y_train.values - pred_train) <= tol).mean() * 100
acc_test  = (np.abs(y_test.values - pred_test) <= tol).mean() * 100

metrics_cb = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "MedianAE", "MaxError", "ExplainedVariance", f"Accuracy@±{tol}"],
    "Train":  [mae_train, rmse_train, r2_train, medae_train, maxerr_train, evs_train, acc_train],
    "Test":   [mae_test, rmse_test, r2_test, medae_test, maxerr_test, evs_test, acc_test],
    "Better if": ["Lower", "Lower", "Higher", "Lower", "Lower", "Higher", "Higher"]
})

print("\n=== CatBoost Baseline Metrics ===")
display(metrics_cb)


# ------------------------------
# 10) SHAP global feature importance (sample)
# ------------------------------
# Use a sample for speed
sample_n = min(1000, len(X_train))
X_shap_sample = X_train.sample(sample_n, random_state=42).copy()

# Make sure code_postal stays string for CatBoost + SHAP
if "code_postal" in X_shap_sample.columns:
    X_shap_sample["code_postal"] = X_shap_sample["code_postal"].astype("string")

# Create explainer
explainer = shap.TreeExplainer(cat_model_baseline)

# SHAP values
shap_values = explainer.shap_values(X_shap_sample)

print("SHAP computed ✅")
print("SHAP values shape:", np.array(shap_values).shape)
print("Sample shape:", X_shap_sample.shape)

# Summary plot (global)
plt.figure()
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.tight_layout()
plt.show()

# Bar plot (global importance)
plt.figure()
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# Global SHAP importance table
shap_importance_df = pd.DataFrame({
    "feature": X_shap_sample.columns.astype(str),
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("Top 20 global SHAP features:")
display(shap_importance_df.head(20))


# ------------------------------
# 11) Predict on your example row (raw categorical values)
# ------------------------------
# Put your example row here EXACTLY
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
        "model": "Pixel 7 Pro",
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
        "delai_Sinistre": 602000
    }
]

X_new = pd.DataFrame(new_rows).copy()

# Ensure all columns exist (same as X)
for c in X.columns:
    if c not in X_new.columns:
        X_new[c] = np.nan

# Keep same column order
X_new = X_new[X.columns].copy()

# Force code_postal categorical
if "code_postal" in X_new.columns:
    X_new["code_postal"] = X_new["code_postal"].astype("string")

# Optional: apply same 0->NaN business rule
for c in zero_to_nan_cols:
    if c in X_new.columns:
        X_new[c] = pd.to_numeric(X_new[c], errors="coerce")
        X_new.loc[X_new[c] == 0, c] = np.nan

# Predict
new_pool = Pool(X_new, cat_features=cat_cols)
pred_new = cat_model_baseline.predict(new_pool)
pred_new_clip = float(np.clip(pred_new, clip_min, clip_max)[0])

print("\n=== Prediction for your example row ===")
print("Predicted value (raw):", float(pred_new[0]))
print("Predicted value (clipped 0..10):", pred_new_clip)
display(X_new)


# ------------------------------
# 12) SHAP local explanation for the example row (why this prediction)
# ------------------------------
shap_values_new = explainer.shap_values(X_new)

# base value (expected value)
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = float(np.array(base_value).reshape(-1)[0])
else:
    base_value = float(base_value)

row_shap = np.array(shap_values_new)[0]
row_vals = X_new.iloc[0]

local_shap_df = pd.DataFrame({
    "feature": X_new.columns.astype(str),
    "feature_value": row_vals.values,
    "shap_value": row_shap,
    "abs_shap_value": np.abs(row_shap)
}).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

local_shap_df["effect"] = np.where(
    local_shap_df["shap_value"] > 0, "augmente la prédiction",
    np.where(local_shap_df["shap_value"] < 0, "diminue la prédiction", "neutre")
)

print("\n=== SHAP local explanation (top 20) ===")
print("Base value (moyenne du modèle):", base_value)
print("Prediction (raw):", float(pred_new[0]))
print("Prediction (clipped):", pred_new_clip)
display(local_shap_df.head(20))

# Waterfall plot (one row)
try:
    shap_exp = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=X_new.iloc[0].values,
        feature_names=X_new.columns.astype(str).tolist()
    )
    plt.figure()
    shap.plots.waterfall(shap_exp, max_display=20, show=False)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Waterfall plot skipped:", e)
    print("Use the local_shap_df table above (same explanation).")
```
