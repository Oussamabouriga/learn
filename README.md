```
# ============================================================
# CATBOOST READY DATA (NO one-hot / NO frequency encoding)
# Minimal transformations:
# - force categorical columns to string + fill missing
# - 0 -> NaN for selected numeric columns (business missing)
# - log1p for selected numeric columns (delays/prices)
# - keep numeric columns numeric
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 1) Copy (keep originals safe)
# -----------------------------
X_train_cb = X_train_no_te.copy()
X_test_cb  = X_test_no_te.copy()

# -----------------------------
# 2) Choose categorical columns (YOU choose)
#    CatBoost needs the list of cat column names
# -----------------------------
cat_cols_cb = [
    "code_postal",   # forced categorical
    # "operating_system",
    # "marque",
    # "model",
    # ...
]
cat_cols_cb = [c for c in cat_cols_cb if c in X_train_cb.columns]

print("Categorical columns for CatBoost:", cat_cols_cb)

# -----------------------------
# 3) 0 -> NaN conversion (YOU choose numeric columns)
#    Only for columns where 0 means "missing" in business logic
# -----------------------------
zero_to_nan_cols_cb = [
    # "some_numeric_col",
    # "some_delay_col",
]
zero_to_nan_cols_cb = [c for c in zero_to_nan_cols_cb if c in X_train_cb.columns]

for c in zero_to_nan_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")
    X_train_cb.loc[X_train_cb[c] == 0, c] = np.nan
    X_test_cb.loc[X_test_cb[c] == 0, c]   = np.nan

print("0->NaN applied on:", zero_to_nan_cols_cb)

# -----------------------------
# 4) Force categorical columns to string + fill missing category
# -----------------------------
for c in cat_cols_cb:
    X_train_cb[c] = X_train_cb[c].astype("string").fillna("__MISSING__")
    X_test_cb[c]  = X_test_cb[c].astype("string").fillna("__MISSING__")

# -----------------------------
# 5) Ensure numeric columns are numeric
# -----------------------------
num_cols_cb = [c for c in X_train_cb.columns if c not in cat_cols_cb]
for c in num_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")

# -----------------------------
# 6) Optional: log1p transform (YOU choose) for skewed numeric columns
#    Typical: delays/prices/amounts
# -----------------------------
log1p_cols_cb = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant",
]
log1p_cols_cb = [c for c in log1p_cols_cb if c in X_train_cb.columns and c in num_cols_cb]

for c in log1p_cols_cb:
    # log1p requires >= 0
    X_train_cb[c] = X_train_cb[c].clip(lower=0)
    X_test_cb[c]  = X_test_cb[c].clip(lower=0)
    X_train_cb[c] = np.log1p(X_train_cb[c])
    X_test_cb[c]  = np.log1p(X_test_cb[c])

print("log1p applied on:", log1p_cols_cb)

# -----------------------------
# 7) Final checks
# -----------------------------
print("\nFinal shapes (CatBoost data):")
print("X_train_cb:", X_train_cb.shape)
print("X_test_cb :", X_test_cb.shape)

non_num = X_train_cb.select_dtypes(exclude=[np.number, "string", "object"]).columns.tolist()
print("Weird dtypes (should be empty):", non_num)

# X_train_cb, X_test_cb are ready for CatBoost
# cat_cols_cb is your cat_features list
```
