```
# ============================================================
# CATBOOST READY DATA (NO one-hot / NO frequency encoding)
# + Robust against column-name mismatches (spaces, hidden chars)
# + Builds a "column reference" you will reuse for test rows
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 0) Helper: normalize column names (prevents PARCOURS_FINAL -> NaN issues)
# -----------------------------
def _normalize_cols(df_):
    df_ = df_.copy()
    df_.columns = (
        df_.columns.astype(str)
        .str.replace("\u00A0", " ", regex=False)  # non-breaking space -> space
        .str.strip()
    )
    return df_

# -----------------------------
# 1) Copy + normalize columns (IMPORTANT)
# -----------------------------
X_train_cb = _normalize_cols(X_train_no_te)
X_test_cb  = _normalize_cols(X_test_no_te)

# Reference column order (used later for example rows)
cb_ref_cols = X_train_cb.columns.tolist()

print("✅ Columns normalized. Example:", cb_ref_cols[:10])

# -----------------------------
# 2) Choose categorical columns (YOU choose)
#    CatBoost needs cat column NAMES (after normalization)
# -----------------------------
cat_cols_cb = [
    "code_postal",   # forced categorical
    # "PARCOURS_FINAL",
    # "PARCOURS_INITIAL",
    # "operating_system",
    # "marque",
    # "model",
    # "garantie",
    # "list_prest",
]
cat_cols_cb = [c for c in cat_cols_cb if c in cb_ref_cols]

print("Categorical columns for CatBoost:", cat_cols_cb)

# -----------------------------
# 3) 0 -> NaN conversion (YOU choose numeric columns)
#    Only where 0 means "missing" in business logic
# -----------------------------
zero_to_nan_cols_cb = [
    # "some_numeric_col",
    # "some_delay_col",
]
zero_to_nan_cols_cb = [c for c in zero_to_nan_cols_cb if c in cb_ref_cols]

for c in zero_to_nan_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")
    X_train_cb.loc[X_train_cb[c] == 0, c] = np.nan
    X_test_cb.loc[X_test_cb[c] == 0, c]   = np.nan

print("0->NaN applied on:", zero_to_nan_cols_cb)

# -----------------------------
# 4) Force categorical to string + fill missing category
# -----------------------------
for c in cat_cols_cb:
    X_train_cb[c] = X_train_cb[c].astype("string").fillna("__MISSING__")
    X_test_cb[c]  = X_test_cb[c].astype("string").fillna("__MISSING__")

# -----------------------------
# 5) Ensure numeric columns are numeric (everything except cat)
# -----------------------------
num_cols_cb = [c for c in cb_ref_cols if c not in cat_cols_cb]
for c in num_cols_cb:
    X_train_cb[c] = pd.to_numeric(X_train_cb[c], errors="coerce")
    X_test_cb[c]  = pd.to_numeric(X_test_cb[c],  errors="coerce")

# -----------------------------
# 6) Optional: log1p transform (YOU choose)
# -----------------------------
log1p_cols_cb = [
    # "delai_declaration",
    # "delai_Sinistre",
    # "montant",
]
log1p_cols_cb = [c for c in log1p_cols_cb if c in num_cols_cb]

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

weird = X_train_cb.select_dtypes(exclude=[np.number, "string", "object"]).columns.tolist()
print("Weird dtypes (should be empty):", weird)

# ✅ Ready:
# - X_train_cb, X_test_cb
# - cat_cols_cb (for CatBoost cat_features)
# - cb_ref_cols (reference columns for future example rows)


# ============================================================
# BUILD A TEST ROW that will NEVER lose values (no accidental NaNs)
# - Normalizes keys
# - Aligns to cb_ref_cols (same order as training)
# - Applies SAME transformations: 0->NaN, cat->string, log1p
# ============================================================

import numpy as np
import pandas as pd

# -----------------------------
# 1) Your raw example row (fill values)
# -----------------------------
test_row_cat_regression_no_te = [{
    "PARCOURS_FINAL": "HORS_APPLE_EE",
    "PARCOURS_INITIAL": "HORS_APPLE_EE",
    "code_postal": "59700",
    # add any other columns you want...
}]

X_new_cb = pd.DataFrame(test_row_cat_regression_no_te).copy()

# -----------------------------
# 2) Normalize the example keys (prevents PARCOURS_FINAL issues)
# -----------------------------
X_new_cb = X_new_cb.copy()
X_new_cb.columns = (
    X_new_cb.columns.astype(str)
    .str.replace("\u00A0", " ", regex=False)
    .str.strip()
)

# -----------------------------
# 3) Add missing columns + align EXACTLY to training reference columns
# -----------------------------
for c in cb_ref_cols:
    if c not in X_new_cb.columns:
        X_new_cb[c] = np.nan

X_new_cb = X_new_cb[cb_ref_cols].copy()

# -----------------------------
# 4) Apply SAME 0->NaN rule (only on chosen columns)
# -----------------------------
for c in zero_to_nan_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")
        X_new_cb.loc[X_new_cb[c] == 0, c] = np.nan

# -----------------------------
# 5) Apply SAME categorical handling
# -----------------------------
for c in cat_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = X_new_cb[c].astype("string").fillna("__MISSING__")

# -----------------------------
# 6) Apply SAME numeric conversion
# -----------------------------
num_cols_cb = [c for c in cb_ref_cols if c not in cat_cols_cb]
for c in num_cols_cb:
    X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")

# -----------------------------
# 7) Apply SAME log1p transform
# -----------------------------
for c in log1p_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = X_new_cb[c].clip(lower=0)
        X_new_cb[c] = np.log1p(X_new_cb[c])

# -----------------------------
# 8) Quick verification (example)
# -----------------------------
print("✅ X_new_cb shape:", X_new_cb.shape)
print("✅ columns match train:", list(X_new_cb.columns) == cb_ref_cols)

# Check that key fields are not accidentally NaN
check_cols = ["PARCOURS_FINAL", "PARCOURS_INITIAL", "code_postal"]
for c in check_cols:
    if c in X_new_cb.columns:
        print(f"{c} =", X_new_cb[c].iloc[0])
```
