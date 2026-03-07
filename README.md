```
# ============================================================
# CATBOOST READY DATA (NO one-hot / NO frequency encoding)
# + force code_postal as categorical
# + auto-detect categorical columns
# + 0 -> NaN for selected numeric columns
# + log1p for selected numeric columns
# + EXAMPLE test row prepared with same transformations
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 1) Copy (keep originals safe)
# -----------------------------
X_train_cb = X_train_no_te.copy()
X_test_cb  = X_test_no_te.copy()

# -----------------------------
# 2) Force code_postal as categorical (string)
# -----------------------------
col_cp = "code_postal"  # change if needed
if col_cp in X_train_cb.columns:
    X_train_cb[col_cp] = X_train_cb[col_cp].astype("string")
    X_test_cb[col_cp]  = X_test_cb[col_cp].astype("string")

# -----------------------------
# 3) Auto-detect categorical columns
# Rule: any column that is NOT numeric => categorical
# -----------------------------
numeric_cols_cb = X_train_cb.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_cb = [c for c in X_train_cb.columns if c not in numeric_cols_cb]

print("Auto-detected categorical columns:", len(cat_cols_cb))
print("Example categorical cols:", cat_cols_cb[:15])

# -----------------------------
# 4) 0 -> NaN conversion (YOU choose numeric columns)
# Only for columns where 0 means "missing" in business logic
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
# 5) Ensure numeric columns are numeric
# -----------------------------
for c in numeric_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")

# -----------------------------
# 6) Optional: log1p transform (YOU choose) for skewed numeric columns
# -----------------------------
log1p_cols_cb = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant",
]
log1p_cols_cb = [c for c in log1p_cols_cb if c in X_train_cb.columns and c in numeric_cols_cb]

for c in log1p_cols_cb:
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

# ============================================================
# 8) EXAMPLE test row (raw) + apply SAME transformations
# ============================================================

test_row_cat_regression_no_te = [{
    # Fill only the columns you have values for
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "code_postal": "59700",
    # "tarif": 19.99,
    # "Age": 43,
    # "delai_declaration": 279000,
    # "delai_Sinistre": 602000,
}]

X_new_cb = pd.DataFrame(test_row_cat_regression_no_te).copy()

# add missing columns and align to train columns
for c in X_train_cb.columns:
    if c not in X_new_cb.columns:
        X_new_cb[c] = np.nan
X_new_cb = X_new_cb[X_train_cb.columns].copy()

# force code_postal categorical
if col_cp in X_new_cb.columns:
    X_new_cb[col_cp] = X_new_cb[col_cp].astype("string")

# 0 -> NaN rule (same list)
for c in zero_to_nan_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")
        X_new_cb.loc[X_new_cb[c] == 0, c] = np.nan

# numeric conversion
for c in numeric_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")

# log1p (same list)
for c in log1p_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = X_new_cb[c].clip(lower=0)
        X_new_cb[c] = np.log1p(X_new_cb[c])

print("\n✅ X_new_cb ready:", X_new_cb.shape)
print("✅ columns match train:", list(X_new_cb.columns) == list(X_train_cb.columns))

# Now you can use:
# train_pool = Pool(X_train_cb, y_train_no_te, cat_features=cat_cols_cb)
# pred = model.predict(X_new_cb)

```
