```
# ============================================================
# CatBoost Regressor — Weighted training for imbalanced target
# (regression sample weights)
# NO functions — clean blocks
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
    "code_postal",   # forced categorical
]

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
# 3) Convert pd.NA -> np.nan
# ==============================
X = X.replace({pd.NA: np.nan})


# ==============================
# 4) Force categorical cols to string + fill missing
# ==============================
for c in cat_cols:
    X[c] = X[c].astype("string")
    X[c] = X[c].fillna("__MISSING__")


# ==============================
# 5) Convert numeric columns to numeric
# ==============================
num_cols = [c for c in X.columns if c not in cat_cols]
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")


# ==============================
# 6) Train/test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)


# ==============================
# 7) Build sample weights for regression imbalance (bin inverse freq)
#    - bins based on quantiles when possible
#    - weights = 1 / freq(bin)
#    - normalize around 1
#    - clip for stability
# ==============================
n_bins = 8
clip_min, clip_max = 0.5, 5.0

# bins on y_train only
try:
    y_bins = pd.qcut(y_train, q=min(n_bins, y_train.nunique()), duplicates="drop")
except Exception:
    y_bins = pd.cut(y_train, bins=min(n_bins, max(2, y_train.nunique())))

bin_counts = y_bins.value_counts()
weights_train = y_bins.map(lambda b: 1.0 / bin_counts[b]).astype(float).values

# normalize and clip
weights_train = weights_train / np.mean(weights_train)
weights_train = np.clip(weights_train, clip_min, clip_max)

print("\nSample weights summary:")
print(pd.Series(weights_train).describe())

print("\nAverage weight by bin:")
tmp = pd.DataFrame({"y": y_train.values, "bin": y_bins.astype(str).values, "w": weights_train})
print(tmp.groupby("bin")["w"].agg(["count", "mean", "min", "max"]).sort_index())


# ==============================
# 8) Build Pools (train with weights)
# ==============================
train_pool_w = Pool(
    X_train, y_train,
    cat_features=cat_cols,
    weight=weights_train
)

test_pool = Pool(
    X_test, y_test,
    cat_features=cat_cols
)


# ==============================
# 9) Train weighted CatBoost model
# ==============================
cat_model_weighted = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2500,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=42,
    eval_metric="RMSE",
    verbose=200,
    allow_writing_files=False
)

cat_model_weighted.fit(
    train_pool_w,
    eval_set=test_pool,
    use_best_model=True,
    early_stopping_rounds=200
)

print("\nCatBoost weighted model trained ✅")


# ==============================
# 10) Metrics + tolerance accuracy (weighted model)
# ==============================
pred_train_w = np.clip(cat_model_weighted.predict(X_train), 0, 10)
pred_test_w  = np.clip(cat_model_weighted.predict(X_test),  0, 10)

mae_train_w = mean_absolute_error(y_train, pred_train_w)
rmse_train_w = np.sqrt(mean_squared_error(y_train, pred_train_w))
r2_train_w = r2_score(y_train, pred_train_w)

mae_test_w = mean_absolute_error(y_test, pred_test_w)
rmse_test_w = np.sqrt(mean_squared_error(y_test, pred_test_w))
r2_test_w = r2_score(y_test, pred_test_w)

print("\n=== CatBoost Weighted Metrics ===")
print(f"Train: MAE={mae_train_w:.4f} | RMSE={rmse_train_w:.4f} | R2={r2_train_w:.4f}")
print(f"Test : MAE={mae_test_w:.4f} | RMSE={rmse_test_w:.4f} | R2={r2_test_w:.4f}")

tol = 1.0
acc_train_w = (np.abs(y_train.values - pred_train_w) <= tol).mean() * 100
acc_test_w  = (np.abs(y_test.values  - pred_test_w)  <= tol).mean() * 100
print(f"\nAccuracy within ±{tol} point (weighted):")
print(f"Train: {acc_train_w:.2f}%")
print(f"Test : {acc_test_w:.2f}%")


# ==============================
# 11) Predict on your example row (same as baseline)
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

num_cols_new = [c for c in X_new.columns if c not in cat_cols]
for c in num_cols_new:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce")

pred_new_w = float(np.clip(cat_model_weighted.predict(X_new)[0], 0, 10))
print("\n=== Example prediction (weighted) ===")
print("Predicted note:", pred_new_w)


# ==============================
# 12) SHAP (global + example) for weighted model
# ==============================
sample_size = min(300, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42).copy()

explainer_w = shap.TreeExplainer(cat_model_weighted)
shap_values_w = explainer_w.shap_values(X_shap)

print("\nSHAP (weighted): global importance")
shap.summary_plot(shap_values_w, X_shap, show=True)

print("\nSHAP (weighted): explanation for the example row")
shap_values_one_w = explainer_w.shap_values(X_new)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_one_w[0],
        base_values=explainer_w.expected_value,
        data=X_new.iloc[0],
        feature_names=X_new.columns
    )
)
plt.show()

```
