```
# ============================================================
# CATBOOST REGRESSOR — BASELINE (Modèle 1)
# CatBoost accepte les variables catégorielles directement.
# Ici on force "code_postal" en catégoriel.
# ============================================================

# ------------------------------
# 1) Install / imports
# ------------------------------
# If needed once:
# !pip install catboost

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 2) Config (editable)
# ------------------------------
target_col = "evaluate_note"
test_size = 0.2
random_state = 42

# Columns where 0 means "missing" (optional)
zero_to_nan_cols = [
    # "delai_de_completude",
    # "montant_indem",
]

# Optional: clip predictions to business range [0..10]
clip_min, clip_max = 0, 10

# ------------------------------
# 3) Prepare dataframe (no encoding)
# ------------------------------
df_cb = df.copy()
df_cb.columns = df_cb.columns.astype(str).str.strip()

# Target numeric + drop missing targets
df_cb[target_col] = pd.to_numeric(df_cb[target_col], errors="coerce")
df_cb = df_cb[df_cb[target_col].notna()].copy()

# Replace selected 0 -> np.nan (optional)
for c in zero_to_nan_cols:
    if c in df_cb.columns:
        df_cb[c] = pd.to_numeric(df_cb[c], errors="coerce")
        df_cb.loc[df_cb[c] == 0, c] = np.nan

# Split X/y
X = df_cb.drop(columns=[target_col]).copy()
y = df_cb[target_col].astype(float).copy()

# ------------------------------
# 4) Force code_postal as categorical (IMPORTANT)
# ------------------------------
if "code_postal" in X.columns:
    # Convert to string => CatBoost treats it as categorical
    X["code_postal"] = X["code_postal"].astype("string")

# Detect categorical columns for CatBoost
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
print("Categorical columns used by CatBoost:", cat_cols)

# ------------------------------
# 5) Train/Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)

# ------------------------------
# 6) CatBoost Pools (important)
# ------------------------------
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool  = Pool(X_test, y_test, cat_features=cat_cols)

# ------------------------------
# 7) Baseline CatBoost model
# ------------------------------
cat_model_baseline = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=random_state,

    # Baseline hyperparameters (safe defaults)
    iterations=2000,
    learning_rate=0.03,
    depth=6,

    l2_leaf_reg=3.0,
    random_strength=1.0,

    # Sampling / regularization
    bootstrap_type="Bernoulli",
    subsample=0.8,

    # Logging
    verbose=200
)

# Train
cat_model_baseline.fit(train_pool)
print("CatBoost baseline trained ✅")

# ------------------------------
# 8) Predict (train + test)
# ------------------------------
pred_train = cat_model_baseline.predict(train_pool)
pred_test  = cat_model_baseline.predict(test_pool)

# Clip to [0,10] if target is note
pred_train = np.clip(pred_train, clip_min, clip_max)
pred_test  = np.clip(pred_test, clip_min, clip_max)

# ------------------------------
# 9) Evaluate (MAE / RMSE / R2)
# ------------------------------
mae_train = mean_absolute_error(y_train, pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
r2_train = r2_score(y_train, pred_train)

mae_test = mean_absolute_error(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
r2_test = r2_score(y_test, pred_test)

metrics_cb_baseline = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Train": [mae_train, rmse_train, r2_train],
    "Test":  [mae_test, rmse_test, r2_test],
    "Better if": ["Lower", "Lower", "Higher"]
})

print("\n=== CatBoost Baseline Results ===")
display(metrics_cb_baseline)

# ------------------------------
# 10) "Accuracy" métier par tolérance (± points)
# ------------------------------
tol = 1.0  # change to 0.5 if you want
acc_train_tol = (np.abs(y_train.values - pred_train) <= tol).mean() * 100
acc_test_tol  = (np.abs(y_test.values - pred_test) <= tol).mean() * 100

print(f"Accuracy@±{tol} (Train): {acc_train_tol:.2f}%")
print(f"Accuracy@±{tol} (Test) : {acc_test_tol:.2f}%")
```
