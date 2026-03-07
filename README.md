```
import numpy as np
import pandas as pd

# ============================================================
# Example row for CatBoost Regressor (NO one-hot / NO frequency encoding)
# You will fill values exactly like in your raw dataframe
# ============================================================

test_row_cat_regression_no_te = [{
    # --- categorical examples (strings / categories)
    "code_postal": "59700",
    # "operating_system": "Android",
    # "marque": "Google",
    # "model": "Pixel 7 Pro",
    # "garantie": "Dommage",
    # "list_prest": "ADVANCED_SWAP",

    # --- numeric examples
    # "Age": 43,
    # "tarif": 19.99,
    # "delai_declaration": 279000,
    # "delai_Sinistre": 602000,
    # "Nbr_ticket_information": 4,
}]

X_new_cb = pd.DataFrame(test_row_cat_regression_no_te).copy()

# ============================================================
# Apply the SAME minimal CatBoost transformations you used
# (must reuse: cat_cols_cb, zero_to_nan_cols_cb, log1p_cols_cb)
# ============================================================

# 1) add missing columns (so it matches training raw columns)
for c in X_train_cb.columns:
    if c not in X_new_cb.columns:
        X_new_cb[c] = np.nan
X_new_cb = X_new_cb[X_train_cb.columns].copy()

# 2) 0 -> NaN (business missing) for selected numeric columns
for c in zero_to_nan_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")
        X_new_cb.loc[X_new_cb[c] == 0, c] = np.nan

# 3) categorical -> string + fill missing
for c in cat_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = X_new_cb[c].astype("string").fillna("__MISSING__")

# 4) numeric -> numeric
num_cols_cb = [c for c in X_new_cb.columns if c not in cat_cols_cb]
for c in num_cols_cb:
    X_new_cb[c] = pd.to_numeric(X_new_cb[c], errors="coerce")

# 5) log1p on selected numeric columns (same as training)
for c in log1p_cols_cb:
    if c in X_new_cb.columns:
        X_new_cb[c] = X_new_cb[c].clip(lower=0)
        X_new_cb[c] = np.log1p(X_new_cb[c])

print("✅ X_new_cb shape:", X_new_cb.shape)
print("✅ columns match train:", list(X_new_cb.columns) == list(X_train_cb.columns))

```
